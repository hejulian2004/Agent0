"""Two CPU/Gloo ranks: one tool execution, shared image state and real TITO."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def _worker(rank, size, rendezvous, directory):
    import torch.distributed as dist
    from agent0_protocol.adapters import QwenModelAdapter
    from agent0_protocol.checkpointed import SessionSpec
    from agent0_protocol.tools import ToolRegistry
    from tools.local_profile import load_config
    from tools.training.checkpointed_rollout import CheckpointedSessionState
    from tools.training.canonical_rollout import schedule

    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=size)
    settings = load_config(Path(__file__).resolve().parents[1], 'local_4090_checkpointed')['checkpointed']
    registry = ToolRegistry()

    def execute(arguments, context):
        path = Path(directory) / 'executions.txt'
        with path.open('a') as stream:
            stream.write(str(rank) + '\n')
        from PIL import Image
        image_path = Path(directory) / 'new_image_state.png'
        Image.new('RGB', (4, 3), 'white').save(image_path)
        context.set_generated_image(image_path)
        return {'success': True, 'value': arguments['value']}

    registry.register({'type': 'function', 'name': 'echo', 'description': 'Echo a value.',
        'strict': True, 'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'}},
            'required': ['value'], 'additionalProperties': False}}, execute)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=False):
            return ''.join(chr(token) for token in ids)

        def convert_tokens_to_ids(self, text):
            return 99999 if text == '<|im_end|>' else 99998

    class Engine:
        def __init__(self):
            self.pending, self.count = {}, 0

        def add_request(self, key, prompt, params, lora_request=None):
            assert lora_request == 'verify-adapter-v0'
            self.count += 1
            text = ('<tool_call>{"name":"echo","arguments":{"value":2}}</tool_call>'
                if self.count == 1 else '{"verdict":"accept","repair_from_step":null,"critique":"",'
                '"evidence":[],"guidance":"","failed_solver_call_ids":[]}')
            ids = [ord(c) for c in text]
            self.pending[key] = SimpleNamespace(request_id=key, finished=True,
                outputs=[SimpleNamespace(token_ids=ids,
                    logprobs=[{token: SimpleNamespace(logprob=-.25)} for token in ids])])

        def step(self):
            output, self.pending = list(self.pending.values()), {}
            return output

        def abort_request(self, keys):
            pass

    tokenizer = Tokenizer()
    rollout = SimpleNamespace(tokenizer=tokenizer, model_adapter=QwenModelAdapter(tokenizer),
        registry=registry, processor=SimpleNamespace(), model_path='cpu-fixture',
        config=SimpleNamespace(response_length=4096))
    spec = SessionSpec('v', 'p', 'verify', [
        {'type': 'message', 'role': 'system', 'content': 'Independent verifier'},
        {'type': 'message', 'role': 'user', 'content': 'Question'}], registry.definitions(), 'g', 'a')
    state = CheckpointedSessionState(rollout, spec, settings, directory, 'verify-adapter-v0', 0)
    schedule([state], Engine(), SimpleNamespace(n=1, max_tokens=4096, logprobs=1), 1,
             dist.group.WORLD, rank)
    assert not state.result.failure_kind, state.result.failure_reason
    raw = state.result.raw_rollout
    assert len(raw['sampled_token_ids']) > 0
    assert all(value == -.25 for value, selected in zip(raw['old_logprobs'], raw['sampling_mask']) if selected)
    assert state.context['current_image_path'].endswith('new_image_state.png')
    gathered = [None] * size
    dist.all_gather_object(gathered, raw)
    assert gathered[0] == gathered[1]
    dist.barrier()
    dist.destroy_process_group()


def test_checkpointed_gloo_once_and_state_sync(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_worker, args=(2, str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    assert (tmp_path / 'executions.txt').read_text().splitlines() == ['0']


def test_scheduler_continues_completed_episode_before_slow_peer():
    from tools.training.canonical_rollout import schedule

    started, completed = [], []

    class State:
        def __init__(self, name):
            self.name = name

        def run(self):
            started.append(self.name)
            yield ('generate', {'name': self.name}, 8)

    class Engine:
        def __init__(self):
            self.jobs = {}

        def add_request(self, key, prompt, params):
            self.jobs[key] = prompt['name']

        def step(self):
            fast = [key for key, name in self.jobs.items() if name != 'slow']
            keys = fast or list(self.jobs)
            return [SimpleNamespace(request_id=key, finished=True) for key in keys]

        def abort_request(self, keys):
            for key in keys:
                self.jobs.pop(key, None)

    def following(state):
        completed.append(state.name)
        return [State('verifier')] if state.name == 'fast' else []

    states = [State('fast'), State('slow')]
    schedule(states, Engine(), SimpleNamespace(), 2, on_complete=following)
    assert completed == ['fast', 'verifier', 'slow']
    assert started == ['fast', 'slow', 'verifier']


def _fsdp_roles_worker(rank, size, rendezvous, directory):
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    from tools.training.role_adapters import RoleAdapters
    from tools.training.role_fsdp_checkpoint import save, load, validate

    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=size)
    torch.manual_seed(17)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(3, 2)

        def forward(self, value):
            return self.q_proj(value)

    policies = RoleAdapters(Tiny(), {'rank': 2, 'alpha': 4, 'target_modules': ['q_proj']}, .01)
    policies.select_fsdp('solve')
    model = FSDP(policies.model, device_id=torch.device('cpu'), use_orig_params=True)
    policies._setup_optimizers(.01)
    schedulers = {mode: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
                  for mode, optimizer in policies.optimizers.items()}
    for mode in ('solve', 'repair', 'verify'):
        policies.select_fsdp(mode)
        model(torch.ones(1, 3)).sum().backward()
        policies.step(mode)
        schedulers['shared'].step()
    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=True, rank0_only=False))
    path = Path(directory) / 'roles'
    path.mkdir(exist_ok=True)
    save(path, model, policies, schedulers, 'cpu-protocol-fingerprint')
    policies.versions = dict.fromkeys(policies.versions, 0)
    for optimizer in policies.optimizers.values():
        optimizer.state.clear()
    load(path, model, policies, schedulers, 'cpu-protocol-fingerprint')
    assert policies.versions == {'shared': 3}
    # The sole optimizer owns the shared adapter parameters.
    for mode, optimizer in policies.optimizers.items():
        observed = torch.tensor(len(optimizer.state))
        dist.all_reduce(observed)
        assert observed.item() > 0
        assert set(optimizer.state) <= set(policies.parameters[mode])
    try:
        validate(path, 'old-protocol')
    except ValueError:
        pass
    else:
        raise AssertionError('old checkpoint fingerprint accepted')
    dist.destroy_process_group()


def test_real_cpu_fsdp_role_optimizers_checkpoint_roundtrip(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_fsdp_roles_worker, args=(2, str(tmp_path / 'fsdp_rendezvous'), str(tmp_path)),
             nprocs=2, join=True)
