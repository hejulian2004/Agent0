import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from agent0_protocol.schema import CanonicalTrajectory
from tools.local_profile import load_config
from tools.training.checkpointed_records import problem_inputs, outcome_scorer


def test_original_problem_images_and_reference_separation():
    from PIL import Image
    trajectory = CanonicalTrajectory('p', [], items=[
        {'type': 'message', 'role': 'system', 'content': 'Solver'},
        {'type': 'message', 'role': 'user', 'content': '<image>\nWhich option? A. 2 B. 4'}],
        metadata={'ground_truth': 'B'})
    original = copy.deepcopy(trajectory.to_dict())
    problem = problem_inputs(trajectory, [Image.new('RGB', (2, 2))])
    assert problem[1]['content'][0]['type'] == 'input_image'
    assert 'ground_truth' not in str(problem)
    assert trajectory.to_dict() == original
    scorer = outcome_scorer('B', 'Which option? A. 2 B. 4')
    assert scorer(SimpleNamespace(trajectory={'items': [
        {'type': 'message', 'role': 'assistant', 'content': 'FINAL_ANSWER: 4'}]})) == 1
    assert scorer(SimpleNamespace(trajectory={'items': [
        {'type': 'message', 'role': 'assistant', 'content': '<answer>B</answer>'}]})) == 0
    with pytest.raises(ValueError, match='marker_mismatch'):
        problem_inputs(trajectory, [])


@pytest.mark.parametrize('train_verifier', [True, False])
def test_all_rollouts_and_references_finish_before_role_updates(monkeypatch, train_verifier):
    from tools.training import checkpointed_trainer as module
    events = []
    settings = load_config(Path(__file__).resolve().parents[1], 'local_4090_checkpointed')['checkpointed']

    class Batch:
        def __init__(self, mode):
            self.meta_info = {'role_mode': mode, 'valid_group_count': 1, 'policy_version': '0'}

        def union(self, other):
            events.append('union:' + self.meta_info['role_mode'])

    class Actor:
        def generate_checkpointed_records(self, batch):
            events.append('rollout:all')
            return SimpleNamespace(non_tensor_batch={'checkpointed_records': []})

        def update_checkpointed_actor(self, **batches):
            events.append('update:joint:' + ','.join(batches))
            return SimpleNamespace(meta_info={'metrics': {'loss': [1.0]}})

    class Reference:
        def compute_ref_log_prob(self, batch):
            events.append('ref:' + batch.meta_info['role_mode'])
            return None

    monkeypatch.setattr(module, 'pack_records', lambda *args:
        ({mode: Batch(mode) if mode != 'verify' or train_verifier else None
          for mode in ('solve', 'repair', 'verify')}, []))
    trainer = SimpleNamespace(config=OmegaConf.create({'actor_rollout_ref': {'rollout': {
        'checkpointed': settings, 'temperature': 1.0}}, 'trainer': {'n_gpus_per_node': 4, 'nnodes': 1}}),
        actor_rollout_wg=Actor(), ref_policy_wg=Reference(), use_reference_policy=True,
        processor=None, tokenizer=SimpleNamespace(pad_token_id=0))
    metrics, _, _ = module.role_step(trainer, None)
    expected = ['rollout:all', 'ref:solve', 'union:solve', 'ref:repair', 'union:repair']
    if train_verifier:
        expected += ['ref:verify', 'union:verify']
    expected += ['update:joint:solve,repair' + (',verify' if train_verifier else '')]
    assert events == expected
    assert metrics['before_accuracy_observed'] == 0


@pytest.mark.parametrize('isolation,suffix,train_verifier', [
    (True, True, True), (False, False, False), (True, False, True)])
def test_complete_native_episode_records_and_role_groups(monkeypatch, tmp_path,
                                                       isolation, suffix, train_verifier):
    import json
    import torch
    from tensordict import TensorDict
    from verl import DataProto
    from agent0_protocol.adapters import QwenModelAdapter
    from agent0_protocol.checkpointed import PROTOCOL, role_prompt
    from agent0_protocol.tools import get_tool_registry
    from tools.training.native_role_lora import NativeRoleLoRA
    from tools.training.checkpointed_records import generate_records, pack_records
    from tools.training.canonical_rollout import object_array

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=False):
            return ''.join(chr(token) for token in ids)

        def convert_tokens_to_ids(self, text):
            return 99999

    class Engine:
        def __init__(self):
            self.pending, self.modes = {}, []

        def add_lora(self, request):
            pass

        def remove_lora(self, identity):
            pass

        def add_request(self, key, prompt, params, lora_request=None):
            assert lora_request.lora_name == 'shared-v0'
            prompt_text = tokenizer.decode(prompt['prompt_token_ids'])
            mode = ('verify' if 'You are an independent Verifier-Repair.' in prompt_text else
                    'repair' if 'Repair Mode:' in prompt_text else 'solve')
            self.modes.append(mode)
            if mode == 'verify':
                wrong = 'FINAL_ANSWER: 3' in tokenizer.decode(prompt['prompt_token_ids'])
                text = json.dumps({'verdict': 'revise' if wrong else 'accept',
                    'repair_from_step': 'S1' if wrong else None, 'critique': 'Recompute.',
                    'evidence': ['E00002'], 'guidance': 'Check addition.', 'failed_solver_call_ids': []})
            else:
                text = 'FINAL_ANSWER: ' + ('3' if mode == 'solve' else '2')
            ids = tokenizer.encode(text)
            self.pending[key] = SimpleNamespace(request_id=key, finished=True,
                outputs=[SimpleNamespace(token_ids=ids,
                    logprobs=[{token: SimpleNamespace(logprob=-.3)} for token in ids])])

        def step(self):
            output, self.pending = list(self.pending.values()), {}
            return output

        def abort_request(self, keys):
            pass

    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, 'get_global_rank', lambda *args: 0)
    settings = load_config(Path(__file__).resolve().parents[1], 'local_4090_checkpointed')['checkpointed']
    settings['output_root'] = str(tmp_path)
    settings['ablation'].update(context_isolation=isolation, suffix_repair=suffix,
                               train_verifier_rl=train_verifier)
    tokenizer, engine = Tokenizer(), Engine()
    processor = SimpleNamespace(tokenizer=tokenizer, image_processor=SimpleNamespace(merge_size=2))
    rollout = SimpleNamespace(model_adapter=QwenModelAdapter(tokenizer), tokenizer=tokenizer,
        processor=processor, registry=get_tool_registry(), model_path='cpu-base',
        inference_engine=SimpleNamespace(llm_engine=engine), sampling_params=SimpleNamespace())
    adapters = NativeRoleLoRA(engine, {'shared': tmp_path / 'shared'},
        request_factory=lambda name, identity, path: SimpleNamespace(lora_name=name, lora_int_id=identity))
    trajectories = [CanonicalTrajectory(str(index), rollout.registry.definitions(), items=[
        {'type': 'message', 'role': 'system', 'content': role_prompt('solve')},
        {'type': 'message', 'role': 'user', 'content': f'Problem {index}: 1+1?'}],
        metadata={'protocol_version': PROTOCOL}).to_dict() for index in range(4)]
    prompts = DataProto(batch=TensorDict({'input_ids': torch.ones((4, 1), dtype=torch.long)}, batch_size=4),
        non_tensor_batch={'canonical_trajectory_json': object_array([json.dumps(t) for t in trajectories]),
                         'reward_model': object_array([{'ground_truth': '2'} for _ in range(4)])})
    adapters.begin_rollout()
    records = generate_records(rollout, prompts, settings, adapters, None, 0)
    adapters.end_rollout()
    batches, flows = pack_records(records.non_tensor_batch['checkpointed_records'], settings, processor, 0, 4)
    assert {s['raw_rollout']['policy_version'] for flow in flows for s in flow['sessions']} == {'0'}
    assert {s['trajectory']['metadata']['adapter'] for flow in flows for s in flow['sessions']} == {'shared'}
    assert {s['trajectory']['metadata']['native_lora_id'] for flow in flows for s in flow['sessions']} == {1}
    assert len(flows) == 32 and all(flow['accepted'] for flow in flows)
    assert len(batches['solve']) == 32 and len(batches['repair']) == 128
    assert 'verify' in engine.modes  # Verify loss can be disabled while every check still runs.
    assert (len(batches['verify']) == 128) if train_verifier else batches['verify'] is None
    assert all(flow['attempts'][0]['correct'] == 0 and flow['attempts'][-1]['correct'] == 1 for flow in flows)
    assert all(flow['transitions'][0]['reward'] == 1 for flow in flows)
    assert all(batch.batch['multiturn_mask'].any() for batch in batches.values() if batch is not None)
