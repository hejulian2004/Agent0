"""Local combined RL command and strict resume provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

from tools.local_profile import load_config, validate_local


def protocol_fingerprint(root, config, model):
    paths = [root / "config.yaml", *sorted((root / "agent0_protocol").glob("*.py")),
             *sorted((root / "sandbox").glob("*.py")), *sorted((root / "tools/training").glob("*.py")),
             root / "tools/local_rl.py", root / "tools/canonical_multimodal.py",
             root / "tools/data_builder/sft_quality.py", root / "verl/prompts/agent0_templates.py",
             root / "verl/workers/fsdp_workers.py", root / "verl/workers/sharding_manager/fsdp_vllm.py",
             root / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
             root / "verl/workers/reward_manager/agent0.py", root / "verl/trainer/ppo/ray_trainer.py"]
    paths.extend(root / config["local_data"][name] for name in ("rl_warmup", "rl_formal", "validation"))
    model_path = Path(model).resolve()
    paths.extend(sorted(model_path.glob("*.json")))
    import copy
    settings = copy.deepcopy(config['rl']['hydra'])
    for key in ('default_local_dir', 'experiment_name', 'resume_mode', 'resume_from_path', 'total_epochs', 'total_training_steps'):
        settings['trainer'].pop(key, None)
    settings['data'].pop('train_files', None)
    schedule = copy.deepcopy(config['rl_schedule'])
    schedule.pop('phase', None)
    payload = {"model": str(model_path), "config": settings,
               "weights": {path.name: [path.stat().st_size, path.stat().st_mtime_ns] for path in sorted(model_path.glob('*.safetensors'))},
               "schedule": schedule,
               "hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    from scripts.launch import flatten, hydra_value, sandbox_environment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase", choices=("full", "warmup", "serc"), default="full")
    parser.add_argument("--resume-dir")
    parser.add_argument("--formal-from-checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    for option in ('batch-size', 'rollout-n', 'concurrency', 'tp', 'pp', 'actor-sp',
                   'max-prompt-length', 'max-response-length', 'max-total-response-length',
                   'max-model-len', 'max-batched-tokens', 'warmup-epochs', 'epochs', 'save-freq', 'workers'):
        parser.add_argument('--' + option, type=int)
    parser.add_argument('--gpu-memory-utilization', type=float)
    args, overrides = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root, "local_4090")
    config["rl_schedule"]["phase"] = args.phase
    hydra = config["rl"]["hydra"]
    mapping = {'batch_size': ('data', 'train_batch_size'),
        'rollout_n': ('actor_rollout_ref', 'rollout', 'n'),
        'concurrency': ('actor_rollout_ref', 'rollout', 'max_num_seqs'),
        'tp': ('actor_rollout_ref', 'rollout', 'tensor_model_parallel_size'),
        'pp': ('actor_rollout_ref', 'rollout', 'pipeline_model_parallel_size'),
        'actor_sp': ('actor_rollout_ref', 'actor', 'ulysses_sequence_parallel_size'),
        'max_prompt_length': ('actor_rollout_ref', 'rollout', 'prompt_length'),
        'max_response_length': ('actor_rollout_ref', 'rollout', 'response_length'),
        'max_total_response_length': ('actor_rollout_ref', 'rollout', 'max_total_response_length'),
        'max_model_len': ('actor_rollout_ref', 'rollout', 'max_model_len'),
        'max_batched_tokens': ('actor_rollout_ref', 'rollout', 'max_num_batched_tokens'),
        'gpu_memory_utilization': ('actor_rollout_ref', 'rollout', 'gpu_memory_utilization'),
        'save_freq': ('trainer', 'save_freq'), 'workers': ('data', 'num_workers')}
    for name, keys in mapping.items():
        value = getattr(args, name)
        if value is not None:
            node = hydra
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = value
    if args.warmup_epochs is not None:
        config['rl_schedule']['warmup_epochs'] = args.warmup_epochs
    if args.epochs is not None:
        config['rl_schedule']['formal_epochs'] = args.epochs
    if overrides:
        from omegaconf import OmegaConf
        hydra = OmegaConf.to_container(OmegaConf.merge(OmegaConf.create(hydra),
            OmegaConf.from_dotlist([item.lstrip('+') for item in overrides])), resolve=True)
        config['rl']['hydra'] = hydra
    rollout = hydra['actor_rollout_ref']['rollout']
    hydra['data']['max_prompt_length'] = rollout['prompt_length']
    hydra['data']['max_response_length'] = rollout['response_length']
    if rollout['tensor_model_parallel_size'] != 4 or rollout['pipeline_model_parallel_size'] != 1:
        raise ValueError('This local HJL adapter supports TP4/PP1; other meshes require separate validation')
    plan = validate_local(config)
    hydra["actor_rollout_ref"]["model"]["path"] = str(Path(args.model).resolve())
    hydra["actor_rollout_ref"]["model"]["use_remove_padding"] = True
    hydra["actor_rollout_ref"]["actor"]["use_remove_padding"] = True
    hydra["data"].setdefault("num_workers", 0)
    hydra["data"]["train_files"] = str(root / config["local_data"]["rl_formal" if args.phase == "serc" else "rl_warmup"])
    hydra["data"]["val_files"] = str(root / config["local_data"]["validation"])
    epochs = (config["rl_schedule"]["warmup_epochs"] if args.phase != "serc" else 0) + (config["rl_schedule"]["formal_epochs"] if args.phase != "warmup" else 0)
    total = (plan["warmup_steps"] if args.phase != "serc" else 0) + (plan["formal_steps"] if args.phase != "warmup" else 0)
    output = Path(args.output or args.resume_dir or root / "checkpoints/rl_local" / datetime.now().strftime("rl_%Y%m%d_%H%M%S"))
    hydra["trainer"].update(total_epochs=epochs, total_training_steps=total, default_local_dir=str(output),
                           resume_mode="disable", experiment_name=output.name)
    hydra["local_schedule"] = {"phase": args.phase, "warmup_epochs": config["rl_schedule"]["warmup_epochs"],
        "formal_epochs": config["rl_schedule"]["formal_epochs"], "formal_data": str(root / config["local_data"]["rl_formal"]),
        "warmup_steps": plan["warmup_steps"], "protocol_fingerprint": "dry-run"}
    if not args.dry_run:
        if not (Path(args.model) / "config.json").is_file():
            raise ValueError("RL model must be an exported HF model directory")
        hydra["local_schedule"]["protocol_fingerprint"] = protocol_fingerprint(root, config, args.model)
        import pyarrow.parquet as pq
        for name in ("rl_warmup", "rl_formal", "validation"):
            rows = pq.read_table(root / config["local_data"][name]).to_pylist()
            if name != "validation" and len(rows) != 200:
                raise ValueError("Local RL phase must have 200 rows")
            from agent0_protocol.schema import CanonicalTrajectory
            from agent0_protocol.local_prompts import render_system_prompt
            from agent0_protocol.tools import get_tool_registry
            for row in rows:
                trajectory = CanonicalTrajectory.from_dict(json.loads(row['canonical_trajectory_json']))
                if trajectory.tools != get_tool_registry().definitions():
                    raise ValueError('RL stored tools conflict with the registry')
                systems = [item for item in trajectory.items if item.get('role') == 'system']
                if len(systems) != 1 or systems[0]['content'] != render_system_prompt():
                    raise ValueError('RL stored system prompt conflicts with HJL')
            if any(not r.get("canonical_trajectory_json") for r in rows):
                raise ValueError("RL data requires canonical prompts; explicitly rebuild local RL")
    if args.resume_dir and args.formal_from_checkpoint:
        raise ValueError('Choose resume-dir or formal-from-checkpoint')
    resume = args.formal_from_checkpoint
    if args.formal_from_checkpoint:
        import fcntl
        lock_path = Path(args.formal_from_checkpoint).parent / '.run.lock'
        if lock_path.exists():
            with lock_path.open('r') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError('Cannot load a checkpoint from an active run') from exc
    if args.resume_dir:
        import fcntl
        lock_path = Path(args.resume_dir) / '.run.lock'
        if lock_path.exists():
            with lock_path.open('r') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError('Cannot resume an active run') from exc
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        resume = find_latest_ckpt_path(args.resume_dir)
        if not resume:
            raise ValueError("No stopped checkpoint found in resume directory")
    if resume:
        info = json.loads((Path(resume) / "local_protocol.json").read_text())
        if info["fingerprint"] != hydra["local_schedule"]["protocol_fingerprint"]:
            raise ValueError("RL protocol/data/model/schedule fingerprint mismatch")
        if args.resume_dir and info.get('mode') != args.phase:
            raise ValueError('Resume phase mode differs from the stopped run')
        if args.formal_from_checkpoint:
            hydra['local_schedule']['phase'] = 'serc'
            hydra['local_schedule']['formal_from_checkpoint'] = True
            hydra['local_schedule']['source_checkpoint_fingerprint'] = info['fingerprint']
            hydra['local_schedule']['source_checkpoint_step'] = info['step']
            hydra['data']['train_files'] = str(root / config['local_data']['rl_formal'])
            hydra['trainer']['total_epochs'] = config['rl_schedule']['formal_epochs']
            hydra['trainer']['total_training_steps'] = info['step'] + plan['formal_steps']
        hydra["trainer"].update(resume_mode="resume_path", resume_from_path=str(resume))
    command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-name", "agent0_trainer"]
    command.extend("++" + key + "=" + hydra_value(value) for key, value in flatten("", hydra))

    if args.dry_run or args.preflight_only:
        import shlex
        print(json.dumps({"steps": total, "schedule": plan, "phase": args.phase}, indent=2))
        print(shlex.join(command))
        return
    if not args.resume_dir:
        output.mkdir(parents=True, exist_ok=False)
        (output / "README.md").write_text("Local RL launched; status requires inspection.\n\n" + json.dumps(hydra, indent=2) +
            "\nBF16 LoRA16/64, FSDP SP4; TP4 PP1; prompt8192; generation4096; response30720; model40960; batch4 n8; concurrency32.\n")
    import os
    import scripts.launch as launch
    launch.DRY_RUN = False
    env = sandbox_environment(config)
    env.update(os.environ)
    from tools.training.launch_logged import run
    raise SystemExit(run(command, output / "launch.log", env=env))


if __name__ == "__main__":
    main()
