import json
from types import SimpleNamespace

from agent0_protocol.checkpointed import SessionSpec
from agent0_protocol.checkpointed_runtime import CheckpointedResponsesRunner
from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
from agent0_protocol.tools import ToolRegistry
from tools.local_profile import load_config
from pathlib import Path


def test_verifier_multiple_real_tool_calls_private_context(tmp_path):
    settings = load_config(Path(__file__).resolve().parents[1], 'local_4090_checkpointed')['checkpointed']
    contexts = []
    registry = ToolRegistry()
    registry.register({'type': 'function', 'name': 'echo', 'description': 'Return input.',
        'parameters': {'type': 'object', 'properties': {'value': {'type': 'integer'}},
                       'required': ['value'], 'additionalProperties': False}, 'strict': True},
        lambda args, context: contexts.append(context['sft_output_root']) or {'success': True, 'value': args['value']})

    class Responses:
        def __init__(self):
            self.calls = []

        def create(self, **request):
            self.calls.append(request)
            step = len(self.calls)
            if step < 3:
                items = [{'type': 'function_call', 'call_id': str(step), 'name': 'echo',
                          'arguments': json.dumps({'value': step})}]
            else:
                items = [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': json.dumps({
                    'verdict': 'accept', 'repair_from_step': None, 'critique': '',
                    'evidence': ['tool:1', 'tool:2'], 'guidance': '', 'failed_solver_call_ids': []})}]}]
            return SimpleNamespace(output=items)

    responses = Responses()
    runtime = ResponsesRuntime(ResponsesConfig('http://localhost/v1', 'fake', 'fake',
        timeout_seconds=300), registry, client=SimpleNamespace(responses=responses), probe_on_init=False)
    spec = SessionSpec('v', 'p', 'verify', [
        {'type': 'message', 'role': 'system', 'content': 'independent verifier'},
        {'type': 'message', 'role': 'user', 'content': 'check original question'}],
        registry.definitions(), 'g', 'a')
    runner = CheckpointedResponsesRunner(runtime, settings, tmp_path,
        lambda trajectory: len(json.dumps(trajectory.items)))
    result = runner.run_session(spec)
    assert not result.failure_kind, result.failure_reason
    assert result.metrics['tool_rounds'] == 2
    assert len(contexts) == 2 and contexts[0] == contexts[1]
    assert str(tmp_path) not in json.dumps(responses.calls[0]['input'])
    assert 'max_output_tokens' not in responses.calls[0]  # Teacher default omits reply cap.
    assert len([i for i in result.trajectory['items'] if i['type'] == 'function_call_output']) == 2


def test_native_adapter_versions_and_routes():
    from tools.training.native_role_lora import NativeRoleLoRA
    import pytest

    class Engine:
        def __init__(self):
            self.loaded, self.removed = [], []

        def add_lora(self, value):
            self.loaded.append(value)

        def remove_lora(self, value):
            self.removed.append(value)

    engine = Engine()
    factory = lambda name, index, path: SimpleNamespace(lora_name=name, lora_int_id=index, lora_path=path)
    paths = dict(solve='/s', repair='/r', verify='/v')
    policies = NativeRoleLoRA(engine, paths, request_factory=factory)
    policies.begin_rollout()
    assert len({policies.request(mode).lora_int_id for mode in paths}) == 3
    with pytest.raises(RuntimeError):
        policies.replace(paths, dict.fromkeys(paths, 1))
    policies.end_rollout()
    assert engine.removed == [1, 2, 3]
    policies.replace(paths, dict.fromkeys(paths, 1))
    policies.begin_rollout()
    assert policies.request('verify').lora_name == 'verify-v1'
    assert policies.request('verify').lora_int_id == 6


def test_role_batch_tito_and_whole_dummy_groups():
    import torch
    from agent0_protocol.checkpointed import SessionResult
    from agent0_protocol.schema import RawRollout
    from tools.training.checkpointed_batch import pack_role

    settings = load_config(Path(__file__).resolve().parents[1], 'local_4090_checkpointed')['checkpointed']
    sessions, states = [], {}
    initial = [{'type': 'message', 'role': 'system', 'content': 'repair'}]
    for index, reward in enumerate((0, 1)):
        spec = SessionSpec(str(index), 'p', 'repair', initial, [], 'g', 'cp')
        raw = RawRollout([11, 12], [11, 12], [21, 22, 23], [-.2, None, -.3],
            [True] * 3, [True] * 5, [True, False, True], [21, 23], policy_version='R:1')
        sessions.append(SessionResult(spec, {}, reward=reward, raw_rollout=raw.to_dict()))
        states[spec.session_id] = SimpleNamespace(pixel_inputs={})
    processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 999))
    data = pack_role(sessions, states, 'repair', settings, processor, 0)
    assert len(data) == 4  # Two real + two dummy rows, a whole group.
    assert data.batch['responses'][:2].tolist() == [[21, 22, 23], [21, 22, 23]]
    assert data.batch['multiturn_mask'].tolist() == [[True, False, True],
        [True, False, True], [False, False, False], [False, False, False]]
    assert torch.allclose(data.batch['old_log_probs'][0], torch.tensor([-.2, 0, -.3]))
    assert data.batch['advantages'][:2, 0].tolist() == [-1, 1]
    assert data.batch['position_ids'].shape == (4, 3, 5)
    sessions[0].reward = None
    assert pack_role(sessions, states, 'repair', settings, processor, 0) is None


def test_tool_infrastructure_classification_preserves_legacy():
    from agent0_protocol.tools import ToolExecutionContext
    registry = ToolRegistry()

    def unavailable(arguments, context):
        raise ConnectionError('tool service unavailable')

    registry.register({'type': 'function', 'name': 'probe', 'description': 'Service fixture.',
        'strict': True, 'parameters': {'type': 'object', 'properties': {},
                                     'additionalProperties': False}}, unavailable)
    call = {'type': 'function_call', 'name': 'probe', 'call_id': 'x', 'arguments': {}}
    legacy = registry.execute(call, ToolExecutionContext())
    isolated = registry.execute(call, ToolExecutionContext({'checkpointed_protocol': True}))
    assert legacy == {'success': False, 'error': 'ConnectionError: tool service unavailable'}
    assert isolated['failure_kind'] == 'infrastructure'


def test_sdk_reasoning_identity_preserved_but_not_sent_to_verifier():
    from agent0_protocol.checkpointed_runtime import response_history
    from agent0_protocol.checkpointed import safe_text_item
    from agent0_protocol.adapters import ResponsesAdapter
    raw = {'type': 'reasoning', 'id': 'rs_real_api_id',
        'summary': [{'type': 'summary_text', 'text': 'Check equation.'}],
        'encrypted_content': 'opaque-real-response-content', 'status': 'completed'}
    item = {'type': 'reasoning', 'summary': raw['summary'], '_responses_item': raw}
    assert response_history([item], ResponsesAdapter()) == [raw]
    snapshot = safe_text_item(item)
    assert '_responses_item' not in snapshot
    assert 'encrypted_content' not in snapshot
