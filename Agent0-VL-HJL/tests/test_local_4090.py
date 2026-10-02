import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from tools.local_profile import load_config, validate_local
from tools.data_builder.local_data import canonical_rl_row, reset
from tools.data_builder.multisource_selection import proportional_quotas, _select_rows
from agent0_protocol.schema import CanonicalTrajectory, ProtocolError
from agent0_protocol.adapters import QwenModelAdapter
from agent0_protocol.tools import get_tool_registry
from tools.training.ulysses_sdpa import packed_sdpa
from tools.training.local_schedule import PhaseLoader

ROOT = Path(__file__).resolve().parents[1]


def test_latest_actual_profile_and_generic_unchanged():
    generic = load_config(ROOT)
    local = load_config(ROOT, "local_4090")
    assert "sft_local" not in generic
    assert validate_local(local) == {"sft_steps": 750, "warmup_steps": 100, "formal_steps": 50,
                                    "trajectories_per_step": 32}
    r = local["rl"]["hydra"]["actor_rollout_ref"]
    assert r["actor"]["ulysses_sequence_parallel_size"] == 4
    assert r["rollout"]["prompt_length"] == 8192
    assert r["rollout"]["response_length"] == 4096
    assert r["rollout"]["max_num_seqs"] == 32
    assert r["rollout"]["max_model_len"] == 40960
    assert r["rollout"]["max_num_batched_tokens"] == 32768
    assert r["rollout"]["gpu_memory_utilization"] == .96


def test_canonical_rl_retains_references_splits_and_images():
    original = {"prompt": [{"role": "user", "content": "<image>\nQ"}], "images": [{"bytes": b"abc"}],
        "reward_model": {"ground_truth": "A"}, "extra_info": {"official_split": "testmini", "source_problem_id": "42"}}
    before = copy.deepcopy(original)
    row = canonical_rl_row(original)
    trajectory = CanonicalTrajectory.from_dict(json.loads(row["canonical_trajectory_json"]))
    assert len(trajectory.tools) == 9
    assert trajectory.items[0]["role"] == "system"
    assert row["extra_info"]["official_split"] == "testmini"
    assert row["images"] == original["images"]
    assert row["reward_model"]["ground_truth"] == "A"
    assert original == before


def test_conflicting_system_rejected():
    with pytest.raises(ProtocolError, match="Conflicting"):
        canonical_rl_row({"prompt": [{"role": "system", "content": "different"}], "reward_model": {}})


def test_function_output_render_uses_image_marker_without_base64(tmp_path):
    image = tmp_path / "tool.png"
    Image.new("RGB", (28, 28), "red").save(image)
    text = QwenModelAdapter(None).render([
        {"type": "function_call_output", "call_id": "c", "output": {"success": True, "images": [str(image)]}}
    ], [], generate=False)
    assert text.count("<image>") == 1
    assert "base64" not in text
    assert '"call_id":"c"' in text


def test_packed_sdpa_samples_do_not_attend_each_other():
    torch.manual_seed(7)
    q, k, v = [torch.randn(1, 2, 8, 4, requires_grad=True) for _ in range(3)]
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    result = packed_sdpa(q, k, v, positions)
    reference = torch.cat([torch.nn.functional.scaled_dot_product_attention(
        q[..., i:i+4, :], k[..., i:i+4, :], v[..., i:i+4, :], is_causal=True) for i in (0, 4)], dim=-2)
    torch.testing.assert_close(result, reference)
    result[..., :4, :].sum().backward()
    assert torch.count_nonzero(v.grad[..., 4:, :]) == 0


def test_repeated_image_positions_do_not_split_sample():
    q, k, v = [torch.randn(1, 2, 6, 4) for _ in range(3)]
    actual = packed_sdpa(q, k, v, torch.tensor([0, 1, 2, 2, 2, 3]))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected)


class Loader:
    def __init__(self, rows):
        self.rows, self.cursor = rows, 0
    def __len__(self):
        return len(self.rows)
    def __iter__(self):
        for value in self.rows[self.cursor:]:
            self.cursor += 1
            yield value
        self.cursor = 0
    def state_dict(self):
        return {"cursor": self.cursor}
    def load_state_dict(self, state):
        self.cursor = state["cursor"]


def test_combined_phase_transition_and_mid_epoch_resume():
    loader = PhaseLoader(Loader(["w1", "w2"]), Loader(["f1", "f2"]), 2, "full")
    assert list(loader) == ["w1", "w2"]
    iterator = iter(loader)
    assert next(iterator) == "w1"
    state = loader.state_dict()
    resumed = PhaseLoader(Loader(["w1", "w2"]), Loader(["f1", "f2"]), 2, "full")
    resumed.load_state_dict(state)
    assert list(resumed) == ["w2"]
    assert list(resumed) == ["f1", "f2"]
    assert resumed.current_phase == "serc"


def test_reset_archives_own_outputs_and_retains_candidates(tmp_path):
    config = load_config(ROOT, "local_4090")
    local = tmp_path / config["local_data"]["sft_root"]
    local.mkdir(parents=True)
    candidate = local / "geometry3k_candidates.jsonl"
    candidate.write_text("keep")
    output = local / "geometry3k_generated.jsonl"
    output.write_text("archive")
    archive = Path(reset(tmp_path, config))
    assert candidate.read_text() == "keep"
    assert not output.exists()
    assert (archive / output.relative_to(tmp_path)).read_text() == "archive"


def test_reset_refuses_active_builder(tmp_path):
    import fcntl
    config = load_config(ROOT, "local_4090")
    local = tmp_path / config["local_data"]["sft_root"]
    local.mkdir(parents=True)
    with (local / "geometry3k_generated.jsonl.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ProtocolError, match="active builder"):
            reset(tmp_path, config)


def test_proportional_quotas_keep_all_sources():
    result = proportional_quotas(dict(a=1000, b=3, c=7), 200)
    assert sum(result.values()) == 200
    assert min(result.values()) >= 1


def test_repurposed_testmini_is_not_an_independent_benchmark(tmp_path):
    from tools.data_builder.split_policy import assert_benchmark_allowed
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'future_eval_excluded_sources': ['mathvista']}))
    with pytest.raises(ValueError, match='testmini'):
        assert_benchmark_allowed('mathvista', None, 'testmini', manifest)
    with pytest.raises(ValueError, match='explicit split'):
        assert_benchmark_allowed('mathvista', [{'question': 'unknown split'}], 'test', manifest)
    assert_benchmark_allowed('mathvista', [{'official_split': 'test'}], 'test', manifest)


def test_lo_ra_merge_does_not_mutate_training_weights():
    from verl.workers.sharding_manager.fsdp_vllm import _merge_peft_weights_for_vllm
    prefix = "base_model.model.layer"
    base = torch.zeros(2, 2)
    params = {prefix + ".base_layer.weight": base,
        prefix + ".lora_A.default.weight": torch.ones(1, 2),
        prefix + ".lora_B.default.weight": torch.ones(2, 1)}
    module = SimpleNamespace(peft_config={"default": SimpleNamespace(lora_alpha=4, r=1, fan_in_fan_out=False)})
    result = _merge_peft_weights_for_vllm(params, module)
    torch.testing.assert_close(result[prefix + ".weight"], torch.full((2, 2), 4.))
    assert torch.count_nonzero(base) == 0


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]
    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)
    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 1000, "<|image_pad|>": 1001}.get(token, 1002)


class Engine:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.jobs = {}
        self.generated = []
    def add_request(self, key, prompt, params):
        self.jobs[key] = next(self.texts)
    def step(self):
        outputs = []
        for key, text in list(self.jobs.items()):
            tokens = Tokenizer().encode(text)
            self.generated.extend(tokens)
            result = SimpleNamespace(token_ids=tokens,
                logprobs=[{token: SimpleNamespace(logprob=-.4)} for token in tokens])
            outputs.append(SimpleNamespace(request_id=key, finished=True, outputs=[result]))
            del self.jobs[key]
        return outputs
    def abort_request(self, keys):
        for key in keys:
            self.jobs.pop(key, None)


def make_state(tmp_path):
    from tools.training.canonical_rollout import TrajectoryState
    rollout = SimpleNamespace(tokenizer=Tokenizer(), model_adapter=QwenModelAdapter(Tokenizer()),
        registry=get_tool_registry(), sandbox_timeout=10, max_total_length=30720,
        max_reasoning_steps=8, repair_threshold=.7,
        config=SimpleNamespace(max_model_len=40960, response_length=4096, max_repairs_per_step=6))
    trajectory = CanonicalTrajectory("test", get_tool_registry().definitions(), items=[
        {"type": "message", "role": "system", "content": "solve"},
        {"type": "message", "role": "user", "content": "40+2"}])
    return TrajectoryState(rollout, trajectory, [], [], tmp_path)


def test_local_streaming_repairs_final_plus_failed_call_and_keeps_sampled_tokens(tmp_path, monkeypatch):
    from tools.training import canonical_rollout as rollout
    from vllm import SamplingParams
    monkeypatch.setattr(rollout, "execute_call_batch", lambda registry, calls, context: [
        {"type": "function_call_output", "call_id": call["call_id"], "output": {"success": False, "error": "real failure"}}
        for call in calls])
    verify = json.dumps(dict(score=1, confidence=.9, critique="ok", tool_check=True))
    engine = Engine(['<answer>42</answer><tool_call>{"name":"python_exec","arguments":{"code":"raise ValueError()"}}</tool_call>',
                     verify, '{"action":"NO_CHANGE"}', '<answer>42</answer>', verify])
    state = make_state(tmp_path)
    rollout.schedule([state], engine, SamplingParams(), 1)
    assert state.failure == ""
    assert state.final == '<answer>42</answer>'
    assert state.repairs == 1
    assert [token for token, selected in zip(state.tokens, state.mask) if selected] == engine.generated
    assert all(probability == -.4 for probability, selected in zip(state.logprobs, state.mask) if selected)
    assert state.trajectory.items[state.trajectory.metadata['final_solver_index']]['content'] == '<answer>42</answer>'
    state.trajectory.validate()


def test_local_streaming_six_round_limit_and_no_final_fallback(tmp_path):
    from tools.training.canonical_rollout import schedule
    from vllm import SamplingParams
    texts = []
    verify = json.dumps(dict(score=1, confidence=.1, critique="repair", tool_check=True))
    for round_index in range(7):
        texts.extend(['<answer>42</answer>', verify])
        if round_index < 6:
            texts.append('{"action":"NO_CHANGE"}')
    state = make_state(tmp_path)
    schedule([state], Engine(texts), SamplingParams(), 1)
    assert state.failure == "repair_exhausted"
    assert state.final is None
    assert state.repairs == 6
    assert state.trajectory.metadata["trajectory_failed"]


def test_local_streaming_context_failure_does_not_truncate_actions(tmp_path):
    from tools.training.canonical_rollout import schedule
    from vllm import SamplingParams
    state = make_state(tmp_path)
    state.rollout.max_total_length = 50
    schedule([state], Engine(['x' * 80]), SamplingParams(), 1)
    assert state.failure == "context_exhausted"
    assert state.final is None
    assert state.tokens == []


def test_stream_commits_exact_quota_and_resumes_without_more_requests(tmp_path, monkeypatch):
    from tools.data_builder import sft_stream
    from scripts.build_sft_dataset import SFTTrajectoryBuilder
    from test_sft_migration import runtime_for, message, verifier, initial
    from tools.data_builder import sft_quality
    runtime, _ = runtime_for([[message('<answer>42</answer>')], [verifier()]])
    template = sft_quality.run_serc(runtime, initial(), trajectory_id='fixture')
    sft_quality.judge_answer(runtime, template, {'ground_truth': '42'})
    attempts = []
    def worker(task, index, runtime, builder, minimum):
        attempts.append(index)
        result = copy.deepcopy(template)
        result.trajectory_id = f'fixture_{index}'
        result.metadata.update(source='fixture', sample_hash=str(index))
        return result, False
    monkeypatch.setattr(sft_stream, '_run_task', worker)
    tasks = [{'question': f'question {index}', 'ground_truth': '42'} for index in range(10)]
    output = tmp_path / 'source.jsonl'
    rows, _ = sft_stream.run_stream(tasks, runtime, SFTTrajectoryBuilder(), output, concurrency=1, target_exported=2)
    assert len(rows) == len(attempts) == 2
    state = json.loads(Path(str(output) + '.state.json').read_text())
    assert state['target_reached']
    assert state['completed_indices'] == [0, 1]
    assert state['committed_output_bytes'] == output.stat().st_size
    rows, _ = sft_stream.run_stream(tasks, runtime, SFTTrajectoryBuilder(), output, concurrency=1, target_exported=2)
    assert len(rows) == len(attempts) == 2


def test_qwen_sdpa_patch_matches_original_on_cpu():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from tools.training.ulysses_sdpa import install
    config = Qwen2_5_VLConfig(text_config={'vocab_size': 32, 'hidden_size': 32,
        'intermediate_size': 64, 'num_hidden_layers': 1, 'num_attention_heads': 4,
        'num_key_value_heads': 2, 'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000,
                                                    'mrope_section': [2, 1, 1]}},
        vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
                       'num_heads': 4, 'out_hidden_size': 32, 'fullatt_block_indexes': [0]})
    config._attn_implementation = 'sdpa'
    torch.manual_seed(3)
    original = Qwen2_5_VLForConditionalGeneration(config).cpu().eval()
    patched = copy.deepcopy(original)
    install(patched, 1)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    positions = torch.arange(6).reshape(1, 1, 6).expand(3, 1, 6)
    with torch.no_grad():
        expected = original(input_ids=ids, position_ids=positions, use_cache=False).logits
        actual = patched(input_ids=ids, position_ids=positions, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _sp4_cpu_worker(rank, rendezvous):
    import torch.distributed as distributed
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from tools.training.ulysses_sdpa import install
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group
    torch.set_num_threads(1)
    distributed.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4)
    set_ulysses_sequence_parallel_group(distributed.group.WORLD)
    try:
        config = Qwen2_5_VLConfig(text_config={'vocab_size': 32, 'hidden_size': 32,
            'intermediate_size': 64, 'num_hidden_layers': 1, 'num_attention_heads': 4,
            'num_key_value_heads': 2, 'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000,
                                                        'mrope_section': [2, 1, 1]}},
            vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
                'num_heads': 4, 'out_hidden_size': 32, 'fullatt_block_indexes': [0]},
            image_token_id=11, vision_start_token_id=10, vision_end_token_id=12)
        config._attn_implementation = 'sdpa'
        torch.manual_seed(3)
        original = Qwen2_5_VLForConditionalGeneration(config).cpu().eval()
        patched = copy.deepcopy(original)
        ids = torch.tensor([[1, 10, 11, 12, 3, 4, 5, 6]])
        positions = torch.arange(8).reshape(1, 1, 8).expand(3, 1, 8)
        pixels = torch.zeros(4, 3 * 2 * 14 * 14)
        grid = torch.tensor([[1, 2, 2]])
        with torch.no_grad():
            expected = original(input_ids=ids, position_ids=positions, pixel_values=pixels,
                                image_grid_thw=grid, use_cache=False).logits
            install(patched, 4)
            actual = patched(input_ids=ids[:, 2*rank:2*rank+2], position_ids=positions[..., 2*rank:2*rank+2],
                pixel_values=pixels, image_grid_thw=grid, use_cache=False, _verl_ulysses_sharded_multimodal=True).logits
        torch.testing.assert_close(actual, expected[:, 2*rank:2*rank+2], atol=1e-5, rtol=1e-5)
    finally:
        distributed.destroy_process_group()


def test_sp4_multimodal_collectives_match_full_attention_on_cpu(tmp_path):
    import torch.multiprocessing as multiprocessing
    multiprocessing.spawn(_sp4_cpu_worker, args=(str(tmp_path / 'gloo'),), nprocs=4, join=True)
