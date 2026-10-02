"""One shared-policy update after consuming separately normalized mode groups."""
from contextlib import nullcontext
import math

import torch

from agent0_protocol.checkpointed import MODES
from verl.trainer.ppo.core_algos import compute_policy_loss, agg_loss, kl_penalty


def mode_weights(counts, settings):
    priors = settings['joint_training']['rl_weighting']['priors']
    values = {mode: priors[mode] * math.sqrt(counts.get(mode, 0)) for mode in MODES}
    total = sum(values.values())
    return {mode: value / total if total else 0.0 for mode, value in values.items()}


def update_joint_policy(actor, batches, weights, dp_size=1):
    """Accumulate all modes with fixed parameters; step once, excluding padding.

    A group's loss is the mean of its completions. Each completion preserves the
    configured token loss aggregation. FSDP averages gradients across DP ranks,
    hence the dp_size factor against the global unpadded completion count.
    """
    config = actor.config
    if int(config.ppo_epochs) != 1:
        raise ValueError('Shared checkpointed updates require ppo_epochs=1')
    actor.actor_module.train()
    actor.actor_optimizer.zero_grad(set_to_none=True)
    metrics, contributed = {}, False
    try:
        for mode in MODES:
            if mode not in batches or not weights[mode]:
                continue
            batch = batches[mode]
            count = batch.meta_info['valid_group_count'] * batch.meta_info['group_size']
            if count <= 0:
                continue
            scale = weights[mode] * dp_size / count
            totals = {'pg_loss': 0.0, 'entropy': 0.0, 'kl_loss': 0.0}
            # One completion at a time preserves group-mean normalization and
            # the existing local profile's micro-batch memory bound.
            for micro in batch.chunk(chunks=len(batch)):
                device = next(actor.actor_module.parameters()).device
                data = {**micro.batch.to(device), **micro.non_tensor_batch}
                entropy, log_prob = actor._forward_micro_batch(
                    data, temperature=batch.meta_info['temperature'])
                mask = data['multiturn_mask']
                if not mask.any():
                    # Every rank participates in the same forward/backward
                    # schedule, including ranks receiving only padding rows.
                    loss = log_prob.sum() * 0.0
                else:
                    pg_loss, _, _, _ = compute_policy_loss(
                        old_log_prob=data['old_log_probs'], log_prob=log_prob,
                        advantages=data['advantages'], response_mask=mask,
                        cliprange=config.clip_ratio,
                        cliprange_low=config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio,
                        cliprange_high=config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio,
                        clip_ratio_c=config.get('clip_ratio_c', 3.0))
                    entropy_loss = agg_loss(entropy, mask, config.loss_agg_mode)
                    loss = pg_loss - config.entropy_coeff * entropy_loss
                    totals['pg_loss'] += float(pg_loss.detach())
                    totals['entropy'] += float(entropy_loss.detach())
                    if config.use_kl_loss:
                        kl_loss = agg_loss(kl_penalty(log_prob, data['ref_log_prob'],
                            config.kl_loss_type), mask, config.loss_agg_mode)
                        loss = loss + config.kl_loss_coef * kl_loss
                        totals['kl_loss'] += float(kl_loss.detach())
                saved = (torch.autograd.graph.save_on_cpu(pin_memory=False)
                         if getattr(actor, 'activation_cpu_offload', False) else nullcontext())
                with saved:
                    (loss * scale).backward()
                contributed = True
            metrics.update({mode + '/actor/' + name: value * dp_size / count
                            for name, value in totals.items()})
        updated = False
        if contributed:
            norm = actor._optimizer_step()
            updated = bool(torch.isfinite(norm).item())
            metrics['actor/grad_norm'] = float(norm.detach())
        metrics['adapter/update_applied'] = int(updated)
        return metrics, updated
    finally:
        actor.actor_optimizer.zero_grad(set_to_none=True)
