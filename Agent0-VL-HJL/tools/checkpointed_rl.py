"""Manual VERL entry for a frozen base and one shared SFT adapter."""
import argparse
import copy
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shlex
import sys

from agent0_protocol.checkpointed import PROTOCOL, ADAPTER_LAYOUT, ADAPTER_FOR_MODE, digest, role_prompt
from tools.local_profile import load_config, validate_local
from tools.training.role_adapters import validate_bundle


def configure(root, config, base_model, bundle):
    config = copy.deepcopy(config)
    cp = config['checkpointed']
    cp['adapter_bundle'] = str(Path(bundle).resolve())
    hydra = config['rl']['hydra']
    limits, sampling = cp['limits'], cp['sampling']
    hydra['actor_rollout_ref']['model']['path'] = str(Path(base_model).resolve())
    hydra['actor_rollout_ref']['model']['use_remove_padding'] = True
    actor = hydra['actor_rollout_ref']['actor']
    actor.update(lora_rank=cp['adapters']['rank'], lora_alpha=cp['adapters']['alpha'],
                 target_modules=cp['adapters']['target_modules'], use_dynamic_bsz=False,
                 ppo_mini_batch_size=sampling['ppo_minibatch_groups'],
                 ppo_micro_batch_size_per_gpu=sampling['micro_batch_size'], use_remove_padding=True)
    rollout = hydra['actor_rollout_ref']['rollout']
    rollout.update(checkpointed=cp, local_protocol=True, n=sampling['solve_n'],
        max_num_seqs=sampling['concurrency'], prompt_length=limits['solve_prompt_tokens'],
        response_length=limits['per_generation_tokens'],
        max_total_response_length=limits['solve_response_tokens'], max_model_len=limits['max_model_tokens'])
    hydra['data'].update(train_batch_size=sampling['problems_per_step'],
        max_prompt_length=limits['solve_prompt_tokens'], max_response_length=limits['solve_response_tokens'],
        filter_overlong_prompts=False, truncation='error')
    hydra['algorithm'].update(adv_estimator='grpo', use_kl_in_reward=False)
    hydra['reward_model'].update(enable=False, reward_manager='checkpointed')
    hydra['trainer'].update(val_before_train=False, test_freq=0)
    return config


def validate_inputs(root, config, base_model, bundle):
    import pyarrow.parquet as pq
    from agent0_protocol.schema import CanonicalTrajectory
    from agent0_protocol.tools import get_tool_registry

    manifest, adapters = validate_bundle(bundle, base_model)
    expected = config['checkpointed']['adapters']
    for mode, path in adapters.items():
        value = json.loads((Path(path) / 'adapter_config.json').read_text())
        if value['r'] != expected['rank'] or value['lora_alpha'] != expected['alpha'] or \
                set(value['target_modules']) != set(expected['target_modules']):
            raise ValueError('SFT adapter config differs from RL config: ' + mode)
    for name in ('rl_warmup', 'rl_formal', 'validation'):
        rows = pq.read_table(root / config['local_data'][name]).to_pylist()
        if name != 'validation' and len(rows) != config['local_data']['rl_rows_per_phase']:
            raise ValueError('RL phase row count mismatch: ' + name)
        for row in rows:
            trajectory = CanonicalTrajectory.from_dict(json.loads(row['canonical_trajectory_json']))
            if row.get('protocol_version') != PROTOCOL or trajectory.metadata.get('protocol_version') != PROTOCOL:
                raise ValueError('Old RL data cannot enter checkpointed training')
            if trajectory.items[0]['content'] != role_prompt('solve') or trajectory.tools != get_tool_registry().definitions():
                raise ValueError('RL prompt/tool registry mismatch')
    return manifest


def main():
    from scripts.launch import flatten, hydra_value, sandbox_environment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='local_4090_checkpointed')
    parser.add_argument('--base-model', '--model', dest='base_model', required=True)
    parser.add_argument('--adapter-bundle', required=True)
    parser.add_argument('--phase', choices=('full', 'warmup', 'serc'), default='full')
    parser.add_argument('--output')
    parser.add_argument('--resume-dir')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = configure(root, load_config(root, args.profile), args.base_model, args.adapter_bundle)
    plan = validate_local(config)
    hydra = config['rl']['hydra']
    schedule = config['rl_schedule']
    epochs = (schedule['warmup_epochs'] if args.phase != 'serc' else 0) + \
             (schedule['formal_epochs'] if args.phase != 'warmup' else 0)
    steps = (plan['warmup_steps'] if args.phase != 'serc' else 0) + \
            (plan['formal_steps'] if args.phase != 'warmup' else 0)
    output = Path(args.output or args.resume_dir or root / config['checkpointed']['rl_output_root'] /
                  datetime.now().strftime('rl_%Y%m%d_%H%M%S')).resolve()
    hydra['data'].update(train_files=str(root / config['local_data'][
        'rl_formal' if args.phase == 'serc' else 'rl_warmup']),
        val_files=str(root / config['local_data']['validation']))
    hydra['trainer'].update(total_epochs=epochs, total_training_steps=steps,
        default_local_dir=str(output), experiment_name=output.name, resume_mode='disable')
    fingerprint = 'dry-run'
    if not args.dry_run:
        manifest = validate_inputs(root, config, args.base_model, args.adapter_bundle)
        from tools.local_rl import protocol_fingerprint
        fingerprint = digest({'code_data_model': protocol_fingerprint(root, config, args.base_model),
                              'bundle': manifest, 'adapter_layout_version': ADAPTER_LAYOUT,
                              'mode_to_adapter': ADAPTER_FOR_MODE, 'entry': Path(__file__).read_text()})
    hydra['actor_rollout_ref']['rollout']['checkpointed']['training_fingerprint'] = fingerprint
    hydra['local_schedule'] = {'phase': args.phase, 'warmup_epochs': schedule['warmup_epochs'],
        'formal_epochs': schedule['formal_epochs'], 'formal_data': str(root / config['local_data']['rl_formal']),
        'warmup_steps': plan['warmup_steps'], 'protocol_fingerprint': fingerprint}
    if args.resume_dir:
        run = Path(args.resume_dir)
        if (run / '.run.lock').exists():
            with (run / '.run.lock').open() as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError('Cannot resume an active run') from exc
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        checkpoint = find_latest_ckpt_path(str(run))
        if not checkpoint:
            raise ValueError('No stopped checkpoint found')
        info = json.loads((Path(checkpoint) / 'local_protocol.json').read_text())
        if info['fingerprint'] != fingerprint or info['mode'] != args.phase:
            raise ValueError('Checkpoint protocol/data/model/schedule mismatch')
        hydra['trainer'].update(resume_mode='resume_path', resume_from_path=checkpoint)
    command = [sys.executable, '-m', 'verl.trainer.main_ppo', '--config-name', 'agent0_trainer']
    command.extend('++' + key + '=' + hydra_value(value) for key, value in flatten('', hydra))
    if args.dry_run or args.preflight_only:
        print(json.dumps({'profile': args.profile, 'protocol_version': PROTOCOL,
            'steps': steps, 'schedule': plan, 'gpu_started': False}, indent=2))
        print(shlex.join(command))
        return
    if not args.resume_dir:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'README.md').write_text('Checkpointed role RL\n\nStatus: manual launch requested.\n' +
            json.dumps(hydra, indent=2) + '\nFrozen base; shared LoRA for Solve/Repair/Verify; one optimizer and scheduler.\n')
    import os
    from tools.training.launch_logged import run
    env = sandbox_environment(config)
    env.update(os.environ)
    raise SystemExit(run(command, output / 'launch.log', env=env))


if __name__ == '__main__':
    main()
