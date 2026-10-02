import copy
import json
from pathlib import Path

import pytest

from agent0_protocol.checkpointed import (
    Episode, SessionResult, SessionSpec, parse_review, transition_reward,
    validate_config, review_specs, make_checkpoint, repair_specs,
)
from agent0_protocol.checkpointed_training import group_advantages, sft_rows
from tools.checkpointed_statistics import summarize
from tools.local_profile import load_config


@pytest.fixture
def settings():
    return load_config(Path(__file__).resolve().parents[1], 'local_4090_checkpointed')['checkpointed']


def message(text, role='assistant'):
    return {'type': 'message', 'role': role, 'content': text}


PROBLEM = [message('system', 'system'), message('question only', 'user')]


def result(spec, text, correct=None):
    items = copy.deepcopy(spec.initial_items) + [message(text)]
    return SessionResult(spec, {'schema_version': 'agent0.responses.v1',
        'trajectory_id': spec.session_id, 'tools': [], 'items': items,
        'metadata': {}, 'rollout': {}}, steps=[{'step_id': 'S1',
        'item_start': len(spec.initial_items), 'item_end': len(items), 'state_before': {}}],
        correct=correct)


def verdict(action='accept', point=None):
    return json.dumps({'verdict': action, 'repair_from_step': point,
        'critique': 'check the calculation', 'guidance': 'recompute independently',
        'evidence': ['E00002'], 'failed_solver_call_ids': []})


def test_configuration(settings):
    validate_config(settings, training=True)
    settings['sampling']['verify_n'] = 1
    with pytest.raises(ValueError, match='multiple_samples'):
        validate_config(settings, training=True)


@pytest.mark.parametrize('action,before,after,expected', [
    ('revise', 0, 1, 1), ('revise', 0, 0, 0), ('revise', 1, 1, 0), ('revise', 1, 0, -1),
    ('accept', 0, 0, -1), ('accept', 1, 1, .2),
    ('uncertain', 0, 1, 1), ('uncertain', 1, 1, 0)])
def test_outcome_rewards(settings, action, before, after, expected):
    spec = SessionSpec('v', 'p', 'verify', PROBLEM, [], 'g', 'a')
    output = result(spec, verdict())
    output.verification = {'verdict': action}
    assert transition_reward(before, after, output, settings) == expected
    output.failure_kind = 'infrastructure'
    assert transition_reward(before, after, output, settings) is None


def test_shadow_branches_and_fresh_recheck(settings):
    episode = Episode('p:rollout:0', PROBLEM, [], settings,
        lambda r: int('FINAL_ANSWER: correct' in r.trajectory['items'][-1]['content']))
    generator = episode.run()
    specs = next(generator)
    specs = generator.send([result(specs[0], 'FINAL_ANSWER: wrong')])
    assert specs[0].initial_items == specs[1].initial_items
    assert 'Solve Mode: solve the original problem independently' not in json.dumps(specs[0].initial_items)
    assert 'FINAL_ANSWER: wrong' in json.dumps(specs[0].initial_items)
    specs = generator.send([result(s, verdict('revise', 'S1')) for s in specs])
    assert len(specs) == 4  # Both verifier branches actually receive Repair groups.
    for spec in specs:
        assert 'FINAL_ANSWER: wrong' not in json.dumps(spec.initial_items[:-1])
        assert spec.checkpoint
    specs = generator.send([result(s, 'FINAL_ANSWER: correct') for s in specs])
    snapshot = json.dumps(specs[0].initial_items)
    assert 'verification_checkpoint' not in snapshot
    assert 'FINAL_ANSWER: wrong' not in snapshot
    assert 'FINAL_ANSWER: correct' in snapshot
    with pytest.raises(StopIteration):
        generator.send([result(s, verdict()) for s in specs])
    assert episode.accepted
    assert [t['reward'] for t in episode.transitions[:2]] == [1, 1]
    rows = sft_rows(episode.to_dict())
    assert len(rows['solve']) == 0
    assert len(rows['repair']) == 4
    assert len(rows['verify']) == 4


def test_earliest_prefix_and_fabricated_evidence(settings):
    spec = SessionSpec('s', 'p', 'solve', PROBLEM, [], 'g', 'a')
    solver = result(spec, 'first step')
    solver.trajectory['items'].append(message('discard this suffix'))
    solver.steps.append({'step_id': 'S2', 'item_start': 3, 'item_end': 4,
                         'state_before': {'current_image_path': '/private/step1.png'}})
    current = {'session_id': 's', 'attempt_id': 'before',
               'trajectory': solver.trajectory, 'steps': solver.steps}
    specs, evidence, _ = review_specs('p', PROBLEM, [], current, 0, 2, 0)
    review = result(specs[0], verdict('revise', 'S2'))
    parsed = parse_review(review.trajectory['items'][-1]['content'], current, review, evidence)
    checkpoint, position = make_checkpoint(review, current, parsed, evidence)
    repairs, _ = repair_specs('p', PROBLEM, [], current, checkpoint, position, 2, True)
    encoded = json.dumps(repairs[0].initial_items)
    assert 'first step' in encoded and 'discard this suffix' not in encoded
    assert repairs[0].initial_image_path == '/private/step1.png'
    parsed['evidence'] = ['imaginary']
    with pytest.raises(ValueError, match='fabricated'):
        parse_review(json.dumps(parsed), current, review, evidence)


def test_missing_rewards_mask_whole_group():
    specs = [SessionSpec(str(i), 'p', 'verify', PROBLEM, [], 'g', 'a') for i in range(2)]
    values = [result(s, verdict()) for s in specs]
    values[0].reward, values[1].reward = 1, None
    assert not any(v['loss_mask'] for v in group_advantages(values, {'verify': 2}).values())
    values[1].reward = 1
    assert all(v['advantage'] == 0 for v in group_advantages(values, {'verify': 2}).values())


def test_statistics_read_only(settings):
    original = copy.deepcopy(settings)
    report = summarize([{'protocol_version': settings['protocol_version'], 'accepted': False,
        'attempts': [{}], 'transitions': [], 'sessions': [{'spec': {'mode': 'verify'},
        'failure_kind': 'model', 'failure_reason': 'response_budget_exhausted',
        'metrics': {'output_tokens': 8192}}]}], settings)
    assert settings == original
    assert report['config_modified'] is False
    assert report['failures']['verify:response_budget_exhausted'] == 1


def test_tool_bonus_unique_successful_only(settings):
    output = result(SessionSpec('v', 'p', 'verify', PROBLEM, [], 'g', 'a'), verdict())
    for call_id, arguments, success in [('a', '{}', True), ('b', '{}', True),
                                        ('c', '{"x":1}', False)]:
        output.trajectory['items'].extend([
            {'type': 'function_call', 'call_id': call_id, 'name': 'python_exec', 'arguments': arguments},
            {'type': 'function_call_output', 'call_id': call_id, 'output': {'success': success}}])
    output.verification = {'verdict': 'revise'}
    assert transition_reward(0, 1, output, settings) == pytest.approx(1.03)
    assert transition_reward(0, 0, output, settings) == 0
    assert transition_reward(1, 0, output, settings) == -1
    assert transition_reward(1, 1, output, settings) == 0
    output.verification = {'verdict': 'accept'}
    assert transition_reward(1, 1, output, settings) == pytest.approx(.215)
    assert transition_reward(0, 0, output, settings) == -1


def test_accept_cannot_claim_repair_success(settings):
    output = result(SessionSpec('v', 'p', 'verify', PROBLEM, [], 'g', 'a'), verdict())
    output.verification = {'verdict': 'accept'}
    with pytest.raises(ValueError, match='accept_cannot_change_outcome'):
        transition_reward(0, 1, output, settings)


def test_mean_repair_credit_masks_unknown_and_preserves_main_branch(settings):
    from agent0_protocol.checkpointed import repair_transition_reward
    output = result(SessionSpec('v', 'p', 'verify', PROBLEM, [], 'g', 'a'), verdict('revise', 'S1'))
    output.verification = {'verdict': 'revise'}
    group = [result(SessionSpec(str(i), 'p', 'repair', PROBLEM, [], 'g', 'a'),
                    'FINAL_ANSWER: x', correct=i) for i in (0, 1)]
    assert repair_transition_reward(0, group, output, settings, 0) == (0, [0, 1], .5)
    settings['rewards']['repair_credit_mode'] = 'mean'
    assert repair_transition_reward(0, group, output, settings, 0) == (.5, [0, 1], .5)
    assert repair_transition_reward(1, group, output, settings, 0) == (-.5, [0, 1], .5)
    group[1].correct = None
    group[1].failure_kind = 'infrastructure'
    assert repair_transition_reward(0, group, output, settings, 0) == (None, [0, None], None)


def test_inherited_solver_tools_do_not_receive_verifier_bonus(settings):
    inherited = PROBLEM + [
        {'type': 'function_call', 'call_id': 's', 'name': 'python_exec', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 's', 'output': {'success': True}}]
    output = result(SessionSpec('v', 'p', 'verify', inherited, [], 'g', 'a'), verdict())
    output.verification = {'verdict': 'accept'}
    assert transition_reward(1, 1, output, settings) == .2


def test_episode_mean_credit_never_selects_best_repair(settings):
    settings['rewards']['repair_credit_mode'] = 'mean'
    episode = Episode('p', PROBLEM, [], settings,
        lambda r: int('FINAL_ANSWER: correct' in r.trajectory['items'][-1]['content']))
    generator = episode.run()
    specs = next(generator)
    specs = generator.send([result(specs[0], 'FINAL_ANSWER: wrong')])
    specs = generator.send([result(s, verdict('revise', 'S1')) for s in specs])
    specs = generator.send([result(s, 'FINAL_ANSWER: ' + ('correct' if i % 2 else 'wrong'))
                            for i, s in enumerate(specs)])
    assert episode.attempts[-1]['correct'] == 0
    assert episode.transitions[0]['reward'] == .5
    assert episode.transitions[0]['after'] == 0
    assert episode.transitions[0]['repair_success_rate'] == .5
    with pytest.raises(StopIteration):
        generator.send([result(s, verdict()) for s in specs])
    assert episode.final['correct'] == 0
    assert episode.transitions[-1]['reward'] == -1


@pytest.mark.parametrize('flag', ['context_isolation', 'suffix_repair', 'train_verifier_rl'])
def test_ablation_flags_are_real_booleans(settings, flag):
    settings['ablation'][flag] = False
    validate_config(settings, training=True)
    settings['ablation'][flag] = 'false'
    with pytest.raises(ValueError, match='invalid_ablation_flag'):
        validate_config(settings, training=True)


def test_ablation_full_restart_and_context_inheritance(settings):
    from agent0_protocol.checkpointed import role_prompt
    settings['ablation']['context_isolation'] = False
    settings['ablation']['suffix_repair'] = False
    validate_config(settings, training=True)
    episode = Episode('p', PROBLEM, [], settings,
        lambda r: int('FINAL_ANSWER: correct' in r.trajectory['items'][-1]['content']))
    generator = episode.run()
    specs = next(generator)
    solver = result(specs[0], 'keep old S1')
    solver.trajectory['items'].append(message('FINAL_ANSWER: wrong'))
    solver.steps.append({'step_id': 'S2', 'item_start': 3, 'item_end': 4, 'state_before': {}})
    solver.trajectory['metadata']['image_state'] = {'current_image_path': '/solver/current.png'}
    specs = generator.send([solver])
    assert specs[0].initial_items[1:4] == solver.trajectory['items'][1:]
    assert specs[0].initial_image_path == '/solver/current.png'
    feedback = json.loads(verdict('revise', 'S2'))
    feedback['evidence'] = ['E00003']
    specs = generator.send([result(s, json.dumps(feedback)) for s in specs])
    for spec in specs:
        assert spec.checkpoint is None and spec.first_step_number == 1
        assert spec.initial_items[0]['content'] == role_prompt('repair', checkpointed=False)
        assert len(spec.initial_items) == 3
        assert 'verification_checkpoint' not in json.dumps(spec.initial_items)
        assert 'keep old S1' not in json.dumps(spec.initial_items)
        assert spec.initial_image_path is None
    specs = generator.send([result(s, 'FINAL_ANSWER: correct') for s in specs])
    assert 'keep old S1' not in json.dumps(specs[0].initial_items)
    with pytest.raises(StopIteration):
        generator.send([result(s, verdict()) for s in specs])
    assert episode.accepted
    from tools.data_builder.checkpointed_quality import audit_flow
    assert audit_flow(episode.to_dict())
    repair_rows = sft_rows(episode.to_dict())['repair']
    assert repair_rows and all(row['metadata']['repair_strategy'] == 'full' for row in repair_rows)
    assert all(row['metadata']['loss_start_item_index'] == 3 for row in repair_rows)


def test_terminal_review_and_repair_exhaustion(settings):
    settings['limits']['max_repair_rounds'] = 1
    episode = Episode('p', PROBLEM, [], settings, lambda _: 0)
    generator = episode.run()
    specs = next(generator)
    specs = generator.send([result(s, 'FINAL_ANSWER: wrong') for s in specs])
    specs = generator.send([result(s, verdict('uncertain', None)) for s in specs])
    specs = generator.send([result(s, 'FINAL_ANSWER: wrong') for s in specs])
    assert len(specs) == 1 and specs[0].trainable is False
    with pytest.raises(StopIteration):
        generator.send([result(s, verdict('revise', 'S1')) for s in specs])
    assert episode.failure_reason == 'repair_budget_exhausted'
    assert episode.sessions[-1].reward is None


def test_real_cpu_adapter_gradient_ownership(settings):
    import torch
    from torch import nn
    from tools.training.role_adapters import RoleAdapters

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(3, 2)

        def forward(self, value):
            return self.q_proj(value)

    config = copy.deepcopy(settings['adapters'])
    config['target_modules'] = ['q_proj']
    policies = RoleAdapters(Tiny(), config, .01)
    before = {name: value.detach().clone() for name, value in policies.model.named_parameters()}
    policies.begin_rollout()
    with pytest.raises(RuntimeError, match='forbidden'):
        policies.step('solve')
    policies.end_rollout()
    policies.select('repair')(torch.ones(1, 3)).sum().backward()
    policies.step('repair')
    changed = [name for name, value in policies.model.named_parameters()
               if not torch.equal(value, before[name])]
    assert changed and all('.repair.' in name for name in changed)
    assert policies.versions == {'solve': 0, 'repair': 1, 'verify': 0}


def test_native_sync_preserves_base_and_exports_separate_roles(settings, tmp_path):
    import torch
    from torch import nn
    from safetensors.torch import load_file
    from tools.training.role_adapters import RoleAdapters
    from tools.training.role_weight_sync import split_weights, save_snapshots

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(3, 2)

    original = Tiny()
    frozen = {name: value.clone() for name, value in original.state_dict().items()}
    config = copy.deepcopy(settings['adapters'])
    config['target_modules'] = ['q_proj']
    policies = RoleAdapters(original, config, .01)
    with torch.no_grad():
        for name, value in policies.model.named_parameters():
            if 'lora_' in name:
                value.fill_(2 if '.repair.' in name else 1)
    base, adapters = split_weights(policies.model.state_dict(), policies.model)
    assert set(base) == set(frozen)
    assert all(torch.equal(base[name], value) for name, value in frozen.items())
    paths = save_snapshots(tmp_path / 'snapshots', adapters, policies.model.peft_config)
    for mode, path in paths.items():
        exported = load_file(str(Path(path) / 'adapter_model.safetensors'))
        assert exported and all('.' + mode + '.' not in name for name in exported)
        assert all(torch.all(value == (2 if mode == 'repair' else 1)) for value in exported.values())
    with pytest.raises(ValueError, match='replace'):
        save_snapshots(tmp_path / 'snapshots', adapters, policies.model.peft_config)


def test_three_adapter_checkpoint_roundtrip(settings, tmp_path):
    import torch
    from torch import nn
    from tools.training.role_adapters import RoleAdapters
    from tools.training.role_checkpoint import save, load_state

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(3, 2)

        def forward(self, value):
            return self.q_proj(value)

    config = copy.deepcopy(settings['adapters'])
    config['target_modules'] = ['q_proj']
    base_module = Tiny()
    frozen_state = copy.deepcopy(base_module.state_dict())
    policies = RoleAdapters(base_module, config, .01)
    policies.select('verify')(torch.ones(1, 3)).sum().backward()
    policies.step('verify')
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text('{"model_type":"test-tiny"}')
    from safetensors.torch import save_file
    save_file(frozen_state, base / 'model.safetensors')
    checkpoint = tmp_path / 'checkpoint'
    save(checkpoint, policies, base, 'fixed-config-data-model')
    restored_base = Tiny()
    restored_base.load_state_dict(frozen_state)
    restored = RoleAdapters.from_bundle(restored_base, base, checkpoint / 'bundle', .01)
    load_state(checkpoint, restored, 'fixed-config-data-model')
    assert restored.versions == {'solve': 0, 'repair': 0, 'verify': 1}
    assert restored.optimizers['verify'].state_dict()['state']
    assert not restored.optimizers['repair'].state_dict()['state']
    for mode in ('solve', 'repair', 'verify'):
        assert all(torch.equal(a, b) for a, b in zip(policies.parameters[mode], restored.parameters[mode]))
    with pytest.raises(ValueError, match='Incompatible'):
        load_state(checkpoint, restored, 'old-data-config')
