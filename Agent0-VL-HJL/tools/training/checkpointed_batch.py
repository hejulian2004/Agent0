"""Pack complete role GRPO groups using actual rollout tokens and image tensors."""
from collections import defaultdict

import numpy as np
import torch
from tensordict import TensorDict

from agent0_protocol.checkpointed import digest
from agent0_protocol.checkpointed_training import group_advantages
from agent0_protocol.schema import ProtocolError, RawRollout
from verl import DataProto
from tools.training.canonical_rollout import object_array


def pack_role(sessions, states, mode, settings, processor, pad_token_id, world_size=4):
    if mode == 'verify' and not settings['ablation']['train_verifier_rl']:
        return None
    from verl.models.transformers.qwen2_vl import get_rope_index
    sizes = {name: settings['sampling'][name + '_n'] for name in ('solve', 'repair', 'verify')}
    advantages = group_advantages(sessions, sizes)
    groups = defaultdict(list)
    for session in sessions:
        if session.spec.mode == mode and session.spec.trainable:
            groups[session.spec.group_id].append(session)
    rows = []
    for group in groups.values():
        if not all(advantages[s.spec.session_id]['loss_mask'] for s in group):
            continue
        if len({digest(s.spec.initial_items) for s in group}) != 1:
            raise ProtocolError('grpo_group_input_mismatch')
        if len({s.raw_rollout['policy_version'] for s in group}) != 1:
            raise ProtocolError('policy_changed_during_grpo_group')
        rows.extend((s, advantages[s.spec.session_id]['advantage'], False) for s in group)
    if not rows:
        return None
    # Pad the tail with whole zero-loss groups; never split a GRPO group.
    group_size = sizes[mode]
    mini = settings['sampling']['ppo_minibatch_groups'] * group_size
    target = len(rows)
    while target % mini or target % world_size:
        target += group_size
    while len(rows) < target:
        rows.extend((rows[0][0], 0.0, True) for _ in range(group_size))
    raw_rows = [RawRollout(**row[0].raw_rollout) for row in rows]
    for raw in raw_rows:
        raw.validate()
    prompt_width = max(len(raw.expanded_prompt_token_ids) for raw in raw_rows)
    response_width = max(max(len(raw.response_token_ids) for raw in raw_rows), 1)
    count = len(rows)
    prompts = torch.full((count, prompt_width), pad_token_id, dtype=torch.long)
    responses = torch.full((count, response_width), pad_token_id, dtype=torch.long)
    attention = torch.zeros((count, prompt_width + response_width), dtype=torch.long)
    actions = torch.zeros_like(responses, dtype=torch.bool)
    old_probs, adv = torch.zeros_like(responses, dtype=torch.float32), torch.zeros_like(responses, dtype=torch.float32)
    positions, pixels = [], []
    for index, ((session, advantage, dummy), raw) in enumerate(zip(rows, raw_rows)):
        p, r = len(raw.expanded_prompt_token_ids), len(raw.response_token_ids)
        prompts[index, -p:] = torch.tensor(raw.expanded_prompt_token_ids)
        responses[index, :r] = torch.tensor(raw.response_token_ids)
        attention[index, -p-response_width:prompt_width] = 1
        attention[index, prompt_width:prompt_width+r] = torch.tensor(raw.response_mask)
        actions[index, :r] = torch.tensor(raw.sampling_mask) if not dummy else False
        old_probs[index, :r] = torch.tensor([value or 0.0 for value in raw.old_logprobs])
        adv[index, :r] = advantage
        pixel = states[session.spec.session_id].pixel_inputs
        pixels.append(pixel)
        positions.append(get_rope_index(processor, torch.cat([prompts[index], responses[index]]),
            image_grid_thw=pixel.get('image_grid_thw'), attention_mask=attention[index]))
    batch = TensorDict({'prompts': prompts, 'responses': responses,
        'input_ids': torch.cat([prompts, responses], dim=-1), 'attention_mask': attention,
        'position_ids': torch.stack(positions), 'multiturn_mask': actions,
        'old_log_probs': old_probs, 'advantages': adv}, batch_size=count)
    return DataProto(batch=batch, non_tensor_batch={
        'multi_modal_inputs': object_array(pixels),
        'role_mode': object_array([mode] * count),
        'group_id': object_array([s.spec.group_id for s, _, _ in rows]),
        'session_id': object_array([s.spec.session_id for s, _, _ in rows]),
        'dummy_group': np.array([dummy for _, _, dummy in rows], dtype=bool)},
        meta_info={'role_mode': mode, 'ppo_mini_batch_size': mini,
                   'global_token_num': attention.sum(-1).tolist()})
