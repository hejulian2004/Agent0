"""CPU evidence for the single shared policy and independent mode objectives."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from omegaconf import OmegaConf

from agent0_protocol.checkpointed import (
    ADAPTERS, ADAPTER_FOR_MODE, ADAPTER_LAYOUT, MODES, PROTOCOL,
    Episode, SessionResult, role_prompt, validate_config,
)
from agent0_protocol.schema import CanonicalTrajectory
from tools.local_profile import load_config
from tools.training.role_adapters import RoleAdapters, validate_bundle
from tools.training.checkpointed_update import mode_weights, update_joint_policy
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss


@pytest.fixture
def settings():
    return load_config(Path(__file__).resolve().parents[1], 'local_4090_checkpointed')['checkpointed']


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3, 2)

    def forward(self, x):
        return self.q_proj(x)


def tiny_policies():
    return RoleAdapters(Tiny(), {'rank': 2, 'alpha': 4, 'target_modules': ['q_proj']}, .01)


def test_one_parameter_set_and_optimizer_for_all_modes():
    policies = tiny_policies()
    assert ADAPTERS == ('shared',)
    assert ADAPTER_FOR_MODE == dict.fromkeys(MODES, 'shared')
    assert set(policies.model.peft_config) == set(policies.optimizers) == {'shared'}
    frozen = {n: p.clone() for n, p in policies.model.named_parameters() if 'lora_' not in n}
    # Three CE updates model the three positive SFT modes on the same weights.
    for mode in MODES:
        previous = [p.clone() for p in policies.parameters['shared']]
        logits = policies.select(mode)(torch.tensor([[1., 2., 3.]]))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0])).backward()
        policies.step(mode)
        assert any(not torch.equal(p, old) for p, old in zip(policies.parameters['shared'], previous))
        outputs = [policies.select(other)(torch.ones(1, 3)).detach() for other in MODES]
        assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    assert policies.versions == {'shared': 3}
    assert all(torch.equal(p, frozen[n]) for n, p in policies.model.named_parameters() if n in frozen)


def test_rollout_lock_stale_batch_and_consumption():
    policies = tiny_policies()
    policies.begin_rollout()
    with pytest.raises(RuntimeError, match='during rollout'):
        policies.claim_update('phase', 0)
    policies.end_rollout()
    policies.claim_update('phase', '0')
    with pytest.raises(RuntimeError, match='already consumed'):
        policies.claim_update('phase', 0)
    policies.step('solve')
    with pytest.raises(RuntimeError, match='Stale'):
        policies.claim_update('next', 0)
    state = policies.state_dict()
    restored = tiny_policies()
    restored.load_state_dict(state)
    assert restored.consumed_phases == {'phase'}
    assert restored.versions == {'shared': 1}


def test_sqrt_global_group_weights(settings):
    result = mode_weights({'solve': 4, 'repair': 1, 'verify': 5}, settings)
    denom = 1 + .3 + .2 * 5 ** .5
    assert result == pytest.approx({'solve': 1 / denom, 'repair': .3 / denom,
                                    'verify': .2 * 5 ** .5 / denom})
    assert mode_weights({'solve': 4}, settings) == {'solve': 1., 'repair': 0., 'verify': 0.}
    assert mode_weights({}, settings) == dict.fromkeys(MODES, 0.)
    assert mode_weights({'solve': 4, 'repair': 1}, settings) == pytest.approx(
        {'solve': 1 / 1.3, 'repair': .3 / 1.3, 'verify': 0})
    settings['joint_training']['rl_weighting']['priors']['verify'] = -1
    with pytest.raises(ValueError, match='invalid_mode_priors'):
        validate_config(settings)


class TinyActor:
    def __init__(self, model, optimizer):
        self.actor_module, self.actor_optimizer = model, optimizer
        self.config = OmegaConf.create(dict(ppo_epochs=1, clip_ratio=.2,
            clip_ratio_low=None, clip_ratio_high=None, entropy_coeff=0.,
            loss_agg_mode='token-mean', use_kl_loss=False, grad_clip=10.))
        self.forward_weights, self.steps = [], 0

    def _forward_micro_batch(self, data, temperature):
        self.forward_weights.append([p.detach().clone() for p in self.actor_module.parameters()])
        logits = self.actor_module(data['features']) / temperature
        log_prob = logits.log_softmax(-1)
        entropy = -(log_prob.exp() * log_prob).sum(-1, keepdim=True).expand_as(log_prob)
        return entropy, log_prob

    def _optimizer_step(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        norm = (self.actor_module.clip_grad_norm_(10.) if isinstance(self.actor_module, FSDP)
                else torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), 10.))
        if torch.isfinite(norm):
            self.actor_optimizer.step()
            self.steps += 1
        return norm


def tiny_batches(model, padding=True):
    batches = {}
    for index, mode in enumerate(MODES):
        groups = 1 if mode == 'repair' else 2
        real = groups * 2
        n = real + (2 if padding and mode == 'solve' else 0)
        features = torch.tensor([[1., float(j + index + 1), 3.] for j in range(n)])
        with torch.no_grad():
            old = model(features).log_softmax(-1)
        mask = torch.ones((n, 2), dtype=torch.bool)
        mask[real:] = False
        advantages = torch.tensor([[1., -1.] if j % 2 == 0 else [-1., 1.] for j in range(n)])
        batches[mode] = DataProto(batch=TensorDict(dict(features=features,
            responses=torch.zeros((n, 2), dtype=torch.long), old_log_probs=old,
            advantages=advantages, multiturn_mask=mask), batch_size=n),
            meta_info=dict(temperature=1., valid_group_count=groups, group_size=2))
    return batches


@pytest.mark.parametrize('use_kl', [False, True])
def test_joint_gradient_matches_weighted_reference_and_steps_once(settings, use_kl):
    torch.manual_seed(13)
    policies = tiny_policies()
    reference = copy.deepcopy(policies.model)
    optimizer = torch.optim.AdamW([p for n, p in reference.named_parameters() if 'lora_' in n], lr=.01)
    batches = tiny_batches(policies.model)
    if use_kl:
        for batch in batches.values():
            batch.batch['ref_log_prob'] = batch.batch['old_log_probs'] + .1
    weights = mode_weights({m: b.meta_info['valid_group_count'] for m, b in batches.items()}, settings)
    loss = 0
    for mode, batch in batches.items():
        count = batch.meta_info['valid_group_count'] * 2
        for row in batch.chunk(chunks=len(batch)):
            if not row.batch['multiturn_mask'].any():
                continue
            log_prob = reference(row.batch['features']).log_softmax(-1)
            pg, *_ = compute_policy_loss(row.batch['old_log_probs'], log_prob,
                row.batch['advantages'], row.batch['multiturn_mask'], .2)
            if use_kl:
                from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
                entropy = -(log_prob.exp() * log_prob).sum(-1, keepdim=True).expand_as(log_prob)
                pg = pg - .02 * agg_loss(entropy, row.batch['multiturn_mask'], 'token-mean')
                pg = pg + .05 * agg_loss(kl_penalty(log_prob, row.batch['ref_log_prob'], 'low_var_kl'),
                                         row.batch['multiturn_mask'], 'token-mean')
            loss = loss + weights[mode] * pg / count
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 10.)
    optimizer.step()
    actor = TinyActor(policies.model, policies.optimizers['shared'])
    if use_kl:
        actor.config.update(use_kl_loss=True, kl_loss_type='low_var_kl', kl_loss_coef=.05, entropy_coeff=.02)
    metrics, updated = update_joint_policy(actor, batches, weights)
    assert updated and actor.steps == 1
    assert all(all(torch.equal(a, b) for a, b in zip(actor.forward_weights[0], snapshot))
               for snapshot in actor.forward_weights)
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(policies.model.parameters(), reference.parameters()))
    assert all(mode + '/actor/pg_loss' in metrics for mode in MODES)
    assert all(p.grad is None for p in policies.model.parameters())


def test_empty_and_nonfinite_joint_updates_do_not_step(settings):
    policies = tiny_policies()
    actor = TinyActor(policies.model, policies.optimizers['shared'])
    _, updated = update_joint_policy(actor, {}, dict.fromkeys(MODES, 0.))
    assert not updated and actor.steps == 0
    batches = tiny_batches(policies.model)
    batches['solve'].batch['features'].fill_(float('nan'))
    before = [p.clone() for p in policies.model.parameters()]
    _, updated = update_joint_policy(actor, {'solve': batches['solve']}, dict(solve=1., repair=0., verify=0.))
    assert not updated and actor.steps == 0
    assert all(torch.equal(a, b) for a, b in zip(policies.model.parameters(), before))


def test_shared_bundle_rejects_old_layout_before_loading_weights(tmp_path):
    from tools.training.role_checkpoint import save
    from safetensors.torch import save_file
    policies = tiny_policies()
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text('{}')
    save_file({'weight': torch.ones(1)}, base / 'model.safetensors')
    path = tmp_path / 'checkpoint'
    save(path, policies, base, 'fingerprint')
    manifest, adapters = validate_bundle(path / 'bundle', base)
    assert set(adapters) == {'shared'}
    assert manifest['adapter_layout_version'] == ADAPTER_LAYOUT
    assert list((path / 'adapters').glob('*/adapter_model.safetensors')) == [
        path / 'adapters/shared/adapter_model.safetensors']
    manifest['adapters'] = dict(solve=adapters['shared'], repair=adapters['shared'], verify=adapters['shared'])
    (path / 'bundle/bundle.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Old checkpoints'):
        validate_bundle(path / 'bundle', base)
    state = policies.state_dict()
    state.pop('adapter_layout_version')
    with pytest.raises(ValueError, match='Incompatible'):
        policies.load_state_dict(state)


def make_flow(settings, initially_correct):
    problem = [{'type': 'message', 'role': 'system', 'content': role_prompt('solve')},
               {'type': 'message', 'role': 'user', 'content': 'What is 1+1?'}]
    ep = Episode('good' if initially_correct else 'fix', problem, [], settings,
                 lambda result: int(result.trajectory['items'][-1]['content'] == 'FINAL_ANSWER: 2'), teacher=True)
    generator = ep.run()
    specs = next(generator)
    while True:
        results = []
        for spec in specs:
            wrong = spec.mode == 'solve' and not initially_correct
            if spec.mode == 'verify':
                has_wrong = 'FINAL_ANSWER: 3' in json.dumps(spec.initial_items)
                text = json.dumps(dict(verdict='revise' if has_wrong else 'accept',
                    repair_from_step='S1' if has_wrong else None, critique='', evidence=[],
                    guidance='Recompute', failed_solver_call_ids=[]))
            else:
                text = 'FINAL_ANSWER: ' + ('3' if wrong else '2')
            trajectory = CanonicalTrajectory(spec.session_id, spec.tools, items=copy.deepcopy(spec.initial_items) + [
                {'type': 'message', 'role': 'assistant', 'content': text}]).to_dict()
            results.append(SessionResult(spec, trajectory, steps=[dict(step_id='S1',
                item_start=len(spec.initial_items), item_end=len(trajectory['items']), state_before={})]))
        try:
            specs = generator.send(results)
        except StopIteration:
            return ep.to_dict()


def test_shared_sft_export_keeps_three_modes_and_repair_boundary(settings, tmp_path):
    from tools.checkpointed_sft import export
    flows = [make_flow(settings, True), make_flow(settings, False)]
    data = tmp_path / 'accepted.jsonl'
    data.write_text(''.join(json.dumps(dict(trajectory={'metadata': {'checkpointed_flow': flow}},
                       data_source='test')) + '\n' for flow in flows))
    data.with_suffix('.manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL, rows=2,
        sha256=hashlib.sha256(data.read_bytes()).hexdigest())))
    counts = export(data, tmp_path / 'roles')
    rows = [json.loads(line) for line in (tmp_path / 'roles/shared.jsonl').read_text().splitlines()]
    assert {r['trajectory']['metadata']['mode'] for r in rows} == set(MODES)
    assert len(rows) == sum(counts.values())
    manifest = json.loads((tmp_path / 'roles/shared.manifest.json').read_text())
    assert manifest['rows_by_mode'] == counts and manifest['adapter'] == 'shared'
    for row in rows:
        t = row['trajectory']
        mode = t['metadata']['mode']
        assert t['items'][0]['content'] == role_prompt(mode)
        if mode == 'repair':
            boundary = t['metadata']['loss_start_item_index']
            assert 'verification_checkpoint' in json.dumps(t['items'][:boundary])
            assert 'FINAL_ANSWER: 3' not in json.dumps(t['items'][boundary:])


def _joint_gloo_worker(rank, size, rendezvous):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=size)
    torch.manual_seed(13)
    policies = tiny_policies()
    reference = copy.deepcopy(policies.model)
    settings = load_config(Path(__file__).resolve().parents[1], 'local_4090_checkpointed')['checkpointed']
    global_batches = tiny_batches(reference)
    weights = mode_weights({m: b.meta_info['valid_group_count'] for m, b in global_batches.items()}, settings)
    reference_optimizer = torch.optim.AdamW([p for n, p in reference.named_parameters() if 'lora_' in n], lr=.01)
    update_joint_policy(TinyActor(reference, reference_optimizer), global_batches, weights)
    model = FSDP(policies.model, device_id=torch.device('cpu'), use_orig_params=True)
    policies._setup_optimizers(.01)
    actor = TinyActor(model, policies.optimizers['shared'])
    local_batches = {m: b.chunk(size)[rank] for m, b in global_batches.items()}
    _, updated = update_joint_policy(actor, local_batches, weights, dp_size=size)
    assert updated and actor.steps == 1
    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=False, rank0_only=False))
    state = model.state_dict()
    assert all(torch.allclose(state[name], value, atol=1e-6)
               for name, value in reference.state_dict().items())
    dist.destroy_process_group()


def test_joint_cpu_fsdp_matches_global_reference(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_joint_gloo_worker, args=(2, str(tmp_path / 'joint_rendezvous')), nprocs=2, join=True)


def test_shared_sft_prepare_and_actual_template_limits(settings, tmp_path, monkeypatch):
    from tools.checkpointed_sft import export
    from tools import local_sft
    import transformers
    import sys
    from types import ModuleType
    flows = [make_flow(settings, True), make_flow(settings, False)]
    data = tmp_path / 'accepted.jsonl'
    data.write_text(''.join(json.dumps(dict(trajectory={'metadata': {'checkpointed_flow': flow}},
                       data_source='test')) + '\n' for flow in flows))
    data.with_suffix('.manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL, rows=2,
        sha256=hashlib.sha256(data.read_bytes()).hexdigest())))
    counts = export(data, tmp_path / 'roles')
    shared = tmp_path / 'roles/shared.jsonl'
    # A deterministic tokenizer/processor exercises validation without weights.
    def render(trajectory, tokenizer):
        return ''.join(json.dumps(item) + '\n' for item in trajectory.items), []
    processor = SimpleNamespace(tokenizer=object())
    class Processor:
        tokenizer = object()
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.tensor([[ord(c) for c in text[0]]])}
    processor = Processor()
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda *a, **k: processor)
    monkeypatch.setattr(local_sft, 'render_with_images', render)
    monkeypatch.setattr(local_sft, 'assistant_labels', lambda ids, tokenizer: list(ids))
    limits = dict.fromkeys(MODES, 20000)
    output = tmp_path / 'swift.jsonl'
    report = local_sft.prepare(shared, 'cpu-fixture', output, 30000, mode_limits=limits)
    assert report['rows_by_mode'] == counts and report['adapter'] == 'shared'
    containers = [json.loads(line) for line in output.read_text().splitlines()]
    trajectories = [json.loads(row['messages'][0]['content']) for row in containers]
    assert all(t['metadata']['sft_max_tokens'] == 20000 for t in trajectories)
    # Execute the real plugin encoder with import-only Swift stubs. Its mask and
    # row-limit code run unchanged; no model, Swift job or GPU is initialized.
    class LengthError(ValueError):
        pass
    for name in ('swift', 'swift.template', 'swift.template.templates', 'swift.template.templates.qwen', 'swift.template.base'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['swift.template'].register_template = lambda *a, **k: None
    sys.modules['swift.template.templates.qwen'].QwenTemplateMeta = lambda *a, **k: None
    sys.modules['swift.template.templates.qwen'].Qwen2_5VLTemplate = object
    sys.modules['swift.template.base'].MaxLengthError = LengthError
    namespace = {}
    plugin = Path(__file__).resolve().parents[1] / 'tools/swift_canonical_plugin.py'
    exec(compile(plugin.read_text(), str(plugin), 'exec'), namespace)
    namespace['render_with_images'] = render
    namespace['assistant_labels'] = lambda ids, tokenizer: list(ids)
    template = namespace['HJLCanonicalTemplate']()
    template.processor, template.tokenizer, template.max_length = processor, object(), 30000
    for trajectory in trajectories:
        inputs = SimpleNamespace(messages=[{'content': json.dumps(trajectory)}])
        encoded = template._encode_truncated(inputs)
        boundary = trajectory['metadata']['loss_start_item_index']
        prefix = CanonicalTrajectory.from_dict(trajectory)
        prefix.items = prefix.items[:boundary]
        n = len(render(prefix, None)[0])
        assert all(label == -100 for label in encoded['labels'][:n])
        assert any(label != -100 for label in encoded['labels'][n:])
        trajectory['metadata']['sft_max_tokens'] = 1
        with pytest.raises(LengthError, match='limit is 1'):
            template._encode_truncated(SimpleNamespace(messages=[{'content': json.dumps(trajectory)}]))
    with pytest.raises(ValueError, match='Overlength'):
        local_sft.prepare(shared, 'cpu-fixture', tmp_path / 'too_short.jsonl',
                          30000, mode_limits=dict.fromkeys(MODES, 1))


def test_shared_fsdp_adapter_export_and_old_sidecar_rejection(tmp_path, monkeypatch):
    from tools import export_fsdp_lora_hf as exporter
    from safetensors.torch import load_file
    from torch.distributed.tensor import Replicate
    path = tmp_path / 'actor'
    path.mkdir()
    (path / 'model_world_size_1_rank_0.pt').touch()
    sidecar = path / 'role_checkpoint.json'
    sidecar.write_text(json.dumps(dict(protocol_version=PROTOCOL, adapter_layout_version=ADAPTER_LAYOUT,
        mode_to_adapter=ADAPTER_FOR_MODE, versions={'shared': 0})))
    class Tensor:
        placements = [Replicate()]
        ndim = 2
        def __init__(self, value):
            self.value, self.shape = value, value.shape
        def to_local(self):
            return self.value
    state = {'base_model.model.q_proj.lora_A.shared.weight': Tensor(torch.ones(2, 3)),
             'base_model.model.q_proj.lora_B.shared.weight': Tensor(torch.ones(2, 2))}
    monkeypatch.setattr(exporter.torch, 'load', lambda *a, **k: state)
    output = exporter.export_adapter(path, tmp_path / 'shared', tmp_path / 'base', 2, 4)
    weights = load_file(str(output / 'adapter_model.safetensors'))
    assert set(weights) == {'base_model.model.q_proj.lora_A.weight', 'base_model.model.q_proj.lora_B.weight'}
    info = json.loads(sidecar.read_text())
    info['versions'] = dict.fromkeys(MODES, 0)
    sidecar.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='Old multi-adapter'):
        exporter.export_adapter(path, tmp_path / 'bad', tmp_path / 'base', 2, 4)


@pytest.mark.parametrize('repair_n', [2, 4, 8])
def test_group_count_excludes_padding_failures_and_completion_count(settings, repair_n):
    from agent0_protocol.checkpointed import SessionSpec
    from agent0_protocol.schema import RawRollout
    from tools.training.checkpointed_batch import pack_role
    settings['sampling']['repair_n'] = repair_n
    initial = [{'type': 'message', 'role': 'system', 'content': role_prompt('repair')}]
    sessions, states = [], {}
    for group, missing, no_actions, size in [('good', False, False, repair_n),
            ('unavailable', True, False, repair_n), ('no_actions', False, True, repair_n),
            ('incomplete', False, False, repair_n - 1)]:
        for j in range(size):
            spec = SessionSpec(f'{group}:{j}', 'p', 'repair', initial, [], group, 'cp')
            raw = RawRollout([11], [11], [21], [-.2] if not no_actions else [None],
                [True], [True, True], [not no_actions], [21] if not no_actions else [], policy_version='0')
            reward = None if missing and j == 0 else float(j % 2)
            sessions.append(SessionResult(spec, {}, reward=reward, raw_rollout=raw.to_dict()))
            states[spec.session_id] = SimpleNamespace(pixel_inputs={})
    processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 999))
    batch = pack_role(sessions, states, 'repair', settings, processor, 0, world_size=4)
    assert batch.meta_info['valid_group_count'] == 1
    assert batch.meta_info['group_size'] == repair_n
    assert sum(~batch.non_tensor_batch['dummy_group']) == repair_n
    assert batch.non_tensor_batch['dummy_group'].any()
    assert mode_weights({'solve': 4, 'repair': batch.meta_info['valid_group_count']}, settings) == pytest.approx(
        {'solve': 1/1.3, 'repair': .3/1.3, 'verify': 0})


def test_native_constructor_rejects_old_three_paths():
    from tools.training.native_role_lora import NativeRoleLoRA
    with pytest.raises(ValueError, match='one shared adapter'):
        NativeRoleLoRA(SimpleNamespace(), dict(solve='/s', repair='/r', verify='/v'))


def test_joint_rpc_dispatch_preserves_independent_mode_batches():
    from verl.single_controller.base.decorator import dispatch_dp_compute_data_proto
    batches = tiny_batches(tiny_policies().model)
    from verl.single_controller.base.worker_group import WorkerGroup
    class CPUGroup(WorkerGroup):
        @property
        def world_size(self):
            return 2
    group = CPUGroup.__new__(CPUGroup)
    _, parts = dispatch_dp_compute_data_proto(group, **batches)
    assert set(parts) == set(MODES)
    assert all(len(parts[mode]) == 2 for mode in MODES)
    assert [len(part) for part in parts['repair']] == [1, 1]
    assert [len(part) for part in parts['solve']] == [3, 3]


def test_actual_joint_worker_rpc_versions_scheduler_and_replay(settings, monkeypatch):
    from contextlib import AbstractContextManager
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    policies = tiny_policies()
    actor = TinyActor(policies.model, policies.optimizers['shared'])
    class Manager(AbstractContextManager):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def preprocess_data(self, data):
            return data
        def postprocess_data(self, data):
            return data
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker._is_actor = True
    worker._is_offload_param = worker._is_offload_optimizer = False
    worker.config = OmegaConf.create({'rollout': {'checkpointed': settings}})
    worker.role_policies = policies
    worker.actor, worker.actor_module_fsdp = actor, policies.model
    worker.role_schedulers = {'shared': torch.optim.lr_scheduler.LambdaLR(policies.optimizers['shared'], lambda step: 1.)}
    worker.device_mesh = SimpleNamespace(size=lambda: 1)
    worker.ulysses_sequence_parallel_size = 1
    worker.ulysses_sharding_manager = Manager()
    batches = tiny_batches(policies.model)
    weights = mode_weights({m: b.meta_info['valid_group_count'] for m, b in batches.items()}, settings)
    for b in batches.values():
        b.meta_info.update(policy_version='0', update_phase_id='phase', joint_weights=weights)
    output = worker.update_checkpointed_actor(**batches)
    assert actor.steps == 1 and policies.versions == {'shared': 1}
    assert worker.role_schedulers['shared'].last_epoch == 1
    assert output.meta_info['metrics']['adapter/shared_version'] == 1
    with pytest.raises(RuntimeError, match='Stale'):
        worker.update_checkpointed_actor(**batches)
    for b in batches.values():
        b.meta_info['policy_version'] = '1'
    with pytest.raises(RuntimeError, match='already consumed'):
        worker.update_checkpointed_actor(**batches)
    worker.config.rollout.checkpointed.ablation.train_verifier_rl = False
    with pytest.raises(ValueError, match='Verifier RL is disabled'):
        worker.update_checkpointed_actor(**batches)
