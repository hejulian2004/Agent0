"""VERL driver loop: finish every role rollout before any adapter update."""
import json
from pathlib import Path
import time

from agent0_protocol.checkpointed import MODES
from tools.checkpointed_statistics import summarize
from tools.training.checkpointed_records import pack_records
from verl import DataProto


def role_step(trainer, batch):
    from omegaconf import OmegaConf
    from verl.trainer.ppo.metric_utils import reduce_metrics

    settings = OmegaConf.to_container(trainer.config.actor_rollout_ref.rollout.checkpointed, resolve=True)
    started = time.monotonic()
    records = trainer.actor_rollout_wg.generate_checkpointed_records(batch)
    generation_seconds = time.monotonic() - started
    batches, flows = pack_records(records.non_tensor_batch['checkpointed_records'], settings,
        trainer.processor, trainer.tokenizer.pad_token_id,
        trainer.config.trainer.n_gpus_per_node * trainer.config.trainer.nnodes)
    metrics = {'timing_s/gen': generation_seconds}
    # Compute all frozen references before entering any role's update.
    for mode in MODES:
        role_batch = batches[mode]
        if role_batch is not None:
            role_batch.meta_info['temperature'] = trainer.config.actor_rollout_ref.rollout.temperature
            if trainer.use_reference_policy:
                role_batch.union(trainer.ref_policy_wg.compute_ref_log_prob(role_batch))
    started = time.monotonic()
    for mode in MODES:
        role_batch = batches[mode]
        if role_batch is None:
            metrics[mode + '/masked_step'] = 1
            continue
        output = trainer.actor_rollout_wg.update_actor(role_batch)
        metrics.update({mode + '/' + name: value for name, value in
                        reduce_metrics(output.meta_info['metrics']).items()})
    metrics['timing_s/update_actor'] = time.monotonic() - started
    report = summarize(flows, settings)
    before = [flow['attempts'][0]['correct'] for flow in flows]
    after = [flow['attempts'][-1]['correct'] for flow in flows]
    for name, values in [('before_accuracy', before), ('after_accuracy', after)]:
        known = [value for value in values if value is not None]
        metrics[name] = sum(known) / len(known) if known else 0.0
        metrics[name + '_observed'] = len(known)
    transitions = report['transitions']
    for before_value in (0, 1):
        total = sum(transitions.get(f'{before_value}->{after_value}', 0) for after_value in (0, 1))
        for after_value in (0, 1):
            key = f'{before_value}->{after_value}'
            metrics['transition/' + key + '_count'] = transitions.get(key, 0)
            metrics['transition/' + key + '_rate'] = transitions.get(key, 0) / total if total else 0.0
    for before_value, action, name in [(0, 'accept', 'false_accept'), (1, 'revise', 'unnecessary_revision')]:
        considered = [transition for flow in flows for transition in flow['transitions']
                      if transition['before'] == before_value]
        count = sum(transition['action'] == action or
                    (action == 'revise' and transition['action'] == 'uncertain') for transition in considered)
        metrics['verifier/' + name + '_count'] = count
        metrics['verifier/' + name + '_rate'] = count / len(considered) if considered else 0.0
        metrics['verifier/' + name + '_observed'] = len(considered)
    rates = [transition['repair_success_rate'] for flow in flows for transition in flow['transitions']
             if transition.get('repair_success_rate') is not None]
    metrics['verifier/repair_success_rate'] = sum(rates) / len(rates) if rates else 0.0
    metrics['verifier/repair_success_rate_observed'] = len(rates)
    for mode in MODES:
        rewards = [s['reward'] for flow in flows for s in flow['sessions']
                   if s['spec']['mode'] == mode and s['spec']['trainable'] and s['reward'] is not None]
        metrics[mode + '/reward'] = sum(rewards) / len(rewards) if rewards else 0.0
        metrics[mode + '/reward_observed'] = len(rewards)
    reviews = [session for flow in flows for session in flow['sessions'] if session['spec']['mode'] == 'verify']
    metrics['average_repair_rounds'] = sum(len(flow['attempts']) - 1 for flow in flows) / len(flows) if flows else 0.0
    calls = sum(session['metrics'].get('tool_calls', 0) for session in reviews)
    metrics['average_verifier_tool_calls'] = calls / len(reviews) if reviews else 0.0
    for key in ('successful_tool_calls', 'duplicate_successful_tool_calls', 'failed_tool_calls'):
        total = sum(session['metrics'].get(key, 0) for session in reviews)
        metrics['verifier/' + key] = total
        metrics['verifier/' + key + '_rate'] = total / calls if calls else 0.0
    metrics['verifier/tool_calls_observed'] = calls
    accepts = [session for session in reviews if (session.get('verification') or {}).get('verdict') == 'accept']
    metrics['tool_free_accept_rate'] = sum(not session['metrics'].get('tool_calls', 0) for session in accepts) / len(accepts) if accepts else 0.0
    metrics['accepts_observed'] = len(accepts)
    return metrics, report, flows


def fit(trainer):
    from omegaconf import OmegaConf
    from verl.utils.tracking import Tracking

    if trainer.use_critic or trainer.use_rm:
        raise ValueError('Checkpointed RL requires role GRPO without a learned reward model')
    logger = Tracking(project_name=trainer.config.trainer.project_name,
        experiment_name=trainer.config.trainer.experiment_name,
        default_backend=trainer.config.trainer.logger,
        config=OmegaConf.to_container(trainer.config, resolve=True))
    trainer.global_steps = 0
    trainer._load_checkpoint()
    root = Path(trainer.config.trainer.default_local_dir)
    root.mkdir(parents=True, exist_ok=True)
    if trainer.global_steps >= trainer.total_training_steps:
        return
    trainer.global_steps += 1
    for _ in range(trainer.config.trainer.total_epochs):
        for batch_dict in trainer.train_dataloader:
            started = time.monotonic()
            metrics, report, flows = role_step(trainer, DataProto.from_single_dict(batch_dict))
            metrics['timing_s/step'] = time.monotonic() - started
            last = trainer.global_steps >= trainer.total_training_steps
            frequency = trainer.config.trainer.save_freq
            if frequency > 0 and (last or trainer.global_steps % frequency == 0):
                trainer._save_checkpoint()
            with (root / 'checkpointed_workbench.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': trainer.global_steps,
                    'phase': trainer.train_dataloader.current_phase,
                    'metrics': metrics, 'statistics': report, 'flows': flows}, ensure_ascii=False) + '\n')
            logger.log(data=metrics, step=trainer.global_steps)
            print('[Checkpointed RL] ' + json.dumps({'step': trainer.global_steps,
                'before_accuracy': metrics['before_accuracy'], 'after_accuracy': metrics['after_accuracy'],
                'transitions': report['transitions']}, ensure_ascii=False), flush=True)
            if last:
                return
            trainer.global_steps += 1
