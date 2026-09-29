"""CPU fake engine check for the local vLLM token/semantic split."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tensordict import TensorDict
from vllm import SamplingParams

from agent0_protocol.adapters import QwenModelAdapter
from agent0_protocol.schema import CanonicalTrajectory
from agent0_protocol.tools import ToolRegistry, get_tool_registry
from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_agent0_rollout_spmd import vLLMAgent0Rollout


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def encode(self, value, add_special_tokens=False):
        return [ord(c) for c in value]

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(i) for i in tokens)


class FakeEngine:
    def __init__(self):
        self.calls = 0
        self.generated = []

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls += 1
        content = ('<tool_call>{"name":"echo","arguments":{"value":2}}</tool_call>'
                   if self.calls == 1 else '2')
        ids = [ord(c) for c in content]
        self.generated.extend(ids)
        sample = SimpleNamespace(token_ids=ids, logprobs=[{i: SimpleNamespace(logprob=-0.25)} for i in ids])
        return [SimpleNamespace(outputs=[sample]) for _ in prompts]


class ChainedImageEngine:
    def __init__(self):
        self.calls = 0
        self.replies = [
            '<tool_call>{"name":"crop_image","arguments":{"bbox":[0,0,3,2]}}</tool_call>',
            '<tool_call>{"name":"visual_analyzer","arguments":{}}</tool_call>',
            'done',
        ]

    def generate(self, prompts, sampling_params, use_tqdm=False):
        content = self.replies[self.calls]
        self.calls += 1
        ids = [ord(c) for c in content]
        sample = SimpleNamespace(token_ids=ids, logprobs=[{i: SimpleNamespace(logprob=-0.25)} for i in ids])
        return [SimpleNamespace(outputs=[sample]) for _ in prompts]


def test_local_rollout_keeps_sampled_ids_and_semantic_items():
    registry = ToolRegistry()
    registry.register({
        "type": "function", "name": "echo", "description": "Return a value.",
        "parameters": {"type": "object", "properties": {"value": {"type": "integer"}},
                       "required": ["value"], "additionalProperties": False}, "strict": True,
    }, lambda args, context: {"success": True, "value": args["value"]})
    tokenizer = CharacterTokenizer()
    rollout = vLLMAgent0Rollout.__new__(vLLMAgent0Rollout)
    rollout.model_path = "fake-qwen"
    rollout.tokenizer = tokenizer
    rollout.registry = registry
    rollout.model_adapter = QwenModelAdapter(tokenizer)
    rollout.inference_engine = FakeEngine()
    rollout.config = OmegaConf.create({"response_length": 256, "max_model_len": 4096,
                                       "n": 1, "prompt_length": 8})
    rollout.sampling_params = SamplingParams(max_tokens=256, logprobs=1, detokenize=False)
    rollout.pad_token_id = 0
    rollout.max_total_length = 1024
    rollout.max_reasoning_steps = 2
    rollout.enable_tool_execution = True
    rollout.enable_step_verification = False
    rollout.enable_self_repair = False
    rollout.max_obs_length = 256
    rollout.sandbox_timeout = 1

    prompt = CanonicalTrajectory("local_1", registry.definitions())
    prompt.append({"type": "message", "role": "user", "content": "echo 2"})
    input_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, ord("x")]])
    batch = TensorDict({"input_ids": input_ids,
                        "attention_mask": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]]),
                        "position_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]])}, batch_size=1)
    source = DataProto(batch=batch,
                       non_tensor_batch={"canonical_trajectory_json": np.array([json.dumps(prompt.to_dict())], dtype=object)},
                       meta_info={"do_sample": True, "policy_version": "policy_1"})
    result = rollout.generate_sequences(source)
    trajectory = CanonicalTrajectory.from_dict(result.non_tensor_batch["canonical_trajectory"][0])
    raw = trajectory.rollout
    assert raw["sampled_token_ids"] == rollout.inference_engine.generated
    assert len(raw["old_logprobs"]) == len(raw["response_token_ids"])
    assert all(value == -0.25 for value in raw["old_logprobs"] if value is not None)
    assert raw["policy_version"] == "policy_1"
    assert [item["type"] for item in trajectory.items] == [
        "message", "function_call", "function_call_output", "message"]
    assert trajectory.items[1]["call_id"] == trajectory.items[2]["call_id"]


def test_local_rollout_passes_and_updates_the_current_image():
    registry = get_tool_registry()
    tokenizer = CharacterTokenizer()
    rollout = vLLMAgent0Rollout.__new__(vLLMAgent0Rollout)
    rollout.model_path = "fake-qwen"
    rollout.tokenizer = tokenizer
    rollout.registry = registry
    rollout.model_adapter = QwenModelAdapter(tokenizer)
    rollout.inference_engine = ChainedImageEngine()
    rollout.config = OmegaConf.create({"response_length": 512, "max_model_len": 4096,
                                       "n": 1, "prompt_length": 8})
    rollout.sampling_params = SamplingParams(max_tokens=512, logprobs=1, detokenize=False)
    rollout.pad_token_id = 0
    rollout.max_total_length = 2048
    rollout.max_reasoning_steps = 3
    rollout.enable_tool_execution = True
    rollout.enable_step_verification = False
    rollout.enable_self_repair = False
    rollout.max_obs_length = 512
    rollout.sandbox_timeout = 1

    prompt = CanonicalTrajectory("local_image_1", registry.definitions())
    prompt.append({"type": "message", "role": "user", "content": "Crop and inspect the image."})
    input_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, ord("x")]])
    batch = TensorDict({"input_ids": input_ids,
                        "attention_mask": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]]),
                        "position_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]])}, batch_size=1)
    source = DataProto(
        batch=batch,
        non_tensor_batch={
            "canonical_trajectory_json": np.array([json.dumps(prompt.to_dict())], dtype=object),
            "multi_modal_data": np.array([{"image": [Image.new("RGB", (8, 6), "green")]}], dtype=object),
        },
        meta_info={"do_sample": True, "policy_version": "policy_1"},
    )

    result = rollout.generate_sequences(source)
    trajectory = CanonicalTrajectory.from_dict(result.non_tensor_batch["canonical_trajectory"][0])
    calls = [item for item in trajectory.items if item["type"] == "function_call"]
    outputs = [item for item in trajectory.items if item["type"] == "function_call_output"]
    assert [call["name"] for call in calls] == ["crop_image", "visual_analyzer"]
    assert outputs[1]["output"]["analysis"]["size"] == [3, 2]


class RepairRolloutEngine:
    def __init__(self):
        self.calls = 0
        self.replies = [
            # 1. Step 1 Solver: bad crop (3x2)
            '<tool_call>{"name":"crop_image","arguments":{"bbox":[0,0,3,2]}}</tool_call>',
            # 2. Step 1 Verifier: low confidence triggers repair
            '{"step_index": 1, "score": -0.8, "confidence": 0.2, "critique": "Crop is invalid"}',
            # 3. Step 1 Repair: PATCH action
            '{"action": "PATCH", "target_step": 1, "patch_type": "tool_call", "new_content": "inspect", "justification": "Inspect original image"}',
            # 4. Step 1 Regenerated Solver: inspect image (should see original 8x6 image after rollback!)
            '<tool_call>{"name":"visual_analyzer","arguments":{}}</tool_call>',
            # 5. Step 2 Solver: final answer
            'The image size is 8 by 6.',
            # 6. Step 2 Verifier: high confidence
            '{"step_index": 2, "score": 1.0, "confidence": 0.95, "critique": "Correct final answer"}',
        ]

    def generate(self, prompts, sampling_params, use_tqdm=False):
        content = self.replies[self.calls]
        self.calls += 1
        ids = [ord(c) for c in content]
        sample = SimpleNamespace(token_ids=ids, logprobs=[{i: SimpleNamespace(logprob=-0.25)} for i in ids])
        return [SimpleNamespace(outputs=[sample]) for _ in prompts]


def test_local_rollout_rolls_back_image_on_repair():
    registry = get_tool_registry()
    tokenizer = CharacterTokenizer()
    rollout = vLLMAgent0Rollout.__new__(vLLMAgent0Rollout)
    rollout.model_path = "fake-qwen"
    rollout.tokenizer = tokenizer
    rollout.registry = registry
    rollout.model_adapter = QwenModelAdapter(tokenizer)
    rollout.inference_engine = RepairRolloutEngine()
    rollout.config = OmegaConf.create({"response_length": 512, "max_model_len": 4096,
                                       "n": 1, "prompt_length": 8})
    rollout.sampling_params = SamplingParams(max_tokens=512, logprobs=1, detokenize=False)
    rollout.pad_token_id = 0
    rollout.max_total_length = 4096
    rollout.max_reasoning_steps = 3
    rollout.enable_tool_execution = True
    rollout.enable_step_verification = True
    rollout.enable_self_repair = True
    rollout.repair_threshold = 0.7
    rollout.max_repairs_per_trajectory = 2
    rollout.max_obs_length = 512
    rollout.sandbox_timeout = 1
    rollout._load_prompt_templates()

    prompt = CanonicalTrajectory("local_repair_1", registry.definitions())
    prompt.append({"type": "message", "role": "user", "content": "Crop or inspect the image."})
    input_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, ord("x")]])
    batch = TensorDict({"input_ids": input_ids,
                        "attention_mask": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1]]),
                        "position_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]])}, batch_size=1)
    source = DataProto(
        batch=batch,
        non_tensor_batch={
            "canonical_trajectory_json": np.array([json.dumps(prompt.to_dict())], dtype=object),
            "multi_modal_data": np.array([{"image": [Image.new("RGB", (8, 6), "green")]}], dtype=object),
        },
        meta_info={"do_sample": True, "policy_version": "policy_1"},
    )

    result = rollout.generate_sequences(source)
    assert result.non_tensor_batch["num_repairs"][0] == 1

    trajectory = CanonicalTrajectory.from_dict(result.non_tensor_batch["canonical_trajectory"][0])
    calls = [item for item in trajectory.items if item["type"] == "function_call"]
    outputs = [item for item in trajectory.items if item["type"] == "function_call_output"]

    # First call was bad crop (3x2), repaired call was visual_analyzer
    assert [call["name"] for call in calls] == ["crop_image", "visual_analyzer"]
    # The first output was crop result
    assert outputs[0]["output"]["image_size"] == [3, 2]
    # Because repair rolled back the image, visual_analyzer saw the original [8, 6] image!
    assert outputs[1]["output"]["analysis"]["size"] == [8, 6]
