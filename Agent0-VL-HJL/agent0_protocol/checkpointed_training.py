"""Outcome-grounded training selection and complete-group GRPO advantages."""
from collections import defaultdict
import copy
import math

from .checkpointed import MODES, PROTOCOL
from .schema import ProtocolError


def sft_rows(flow):
    if flow['protocol_version'] != PROTOCOL:
        raise ProtocolError('old_flow_cannot_be_converted')
    if not flow['accepted'] or flow['final']['correct'] != 1:
        return {mode: [] for mode in MODES}
    transitions = {t['session_id']: t for t in flow['transitions']}
    rows = {mode: [] for mode in MODES}
    for session in flow['sessions']:
        spec = session['spec']
        if session['failure_kind']:
            continue
        if spec['mode'] in ('solve', 'repair'):
            positive = session['correct'] == 1
        else:
            transition = transitions.get(spec['session_id'], {})
            positive = (transition.get('before'), transition.get('after')) == (0, 1) or (
                (transition.get('before'), transition.get('after')) == (1, 1)
                and transition.get('action') == 'accept')
        if positive:
            trajectory = copy.deepcopy(session['trajectory'])
            trajectory['metadata'].update(protocol_version=PROTOCOL,
                mode=spec['mode'], problem_id=flow['problem_id'],
                loss_start_item_index=len(spec['initial_items']),
                checkpoint_id=(spec.get('checkpoint') or {}).get('checkpoint_id'))
            rows[spec['mode']].append(trajectory)
    return rows


def group_advantages(sessions, expected_sizes):
    """Missing/infrastructure results mask the complete group, never become zero labels."""
    groups = defaultdict(list)
    for session in sessions:
        if session.spec.trainable:
            groups[(session.spec.mode, session.spec.group_id)].append(session)
    result = {}
    for (mode, group_id), values in groups.items():
        valid = len(values) == expected_sizes[mode] and all(v.reward is not None for v in values)
        if valid:
            mean = sum(v.reward for v in values) / len(values)
            std = math.sqrt(sum((v.reward - mean) ** 2 for v in values) / len(values))
        for value in values:
            result[value.spec.session_id] = {'loss_mask': valid,
                'advantage': (value.reward - mean) / std if valid and std > 0 else 0.0,
                'group_id': group_id, 'mode': mode}
    return result
