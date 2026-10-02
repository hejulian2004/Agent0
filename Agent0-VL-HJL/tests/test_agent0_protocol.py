"""Responses-only protocol acceptance checks without an external endpoint."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent0_protocol.adapters import QwenModelAdapter, ResponsesAdapter
from agent0_protocol.responses_runtime import CapabilityError, ResponsesConfig, ResponsesRuntime
from agent0_protocol.schema import CanonicalTrajectory, ProtocolError, RawRollout
from agent0_protocol.tools import ToolRegistry, get_tool_registry
from agent0_protocol.verifier import (
    parse_repair_instruction,
    parse_verification_output,
    repair_function,
    retry_function,
    verify_trajectory,
)


class FakeResponse:
    def __init__(self, output):
        self.id = "resp_mock"
        self.output = output


class FakeResponses:
    def __init__(self):
        self.requests = []
        self.counter = 0

    def create(self, **request):
        self.requests.append(request)
        self.counter += 1
        if request.get("tool_choice") == "none":
            return FakeResponse([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}])
        if request.get("tool_choice"):
            return FakeResponse([{"type": "function_call", "call_id": f"probe_{self.counter}", "name": "agent0_capability_probe", "arguments": '{"value":1}'}])
        if not request.get("tools") or isinstance(request["input"], str):
            return FakeResponse([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]}])
        outputs = [entry for entry in request["input"] if isinstance(entry, dict) and entry.get("type") == "function_call_output"]
        if len(outputs) < 2:
            return FakeResponse([{"type": "function_call", "call_id": f"run_{len(outputs)+1}", "name": "echo", "arguments": '{"value":1}'}])
        return FakeResponse([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "2"}]}])


class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=False):
        return self.value

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 1, "<|endoftext|>": 2}[token]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register({
            "type": "function", "name": "echo", "description": "Return an integer.",
            "parameters": {"type": "object", "properties": {"value": {"type": "integer"}},
                           "required": ["value"], "additionalProperties": False}, "strict": True,
        }, lambda args, context: {"success": True, "value": args["value"]})
        self.config = ResponsesConfig("http://localhost:8000/v1", "test", "model", max_retries=0)

    def test_probe_text_image_and_two_tool_rounds(self):
        fake = FakeResponses()
        runtime = ResponsesRuntime(self.config, self.registry, client=SimpleNamespace(responses=fake))
        self.assertEqual(len(fake.requests), 5)
        self.assertEqual(fake.requests[1]["input"][0]["content"][1]["type"], "input_image")
        self.assertEqual(sum(bool(req.get("tool_choice")) for req in fake.requests), 3)
        self.assertTrue(any(any(isinstance(item, dict) and item.get("type") == "function_call_output"
                                 for item in req["input"]) for req in fake.requests[3:]))
        self.assertIs(runtime.registry, self.registry)

    def test_sdk_timeout_retry_configuration_and_probe_failure(self):
        config = ResponsesConfig("http://localhost:8000/v1", "test", "model",
                                 timeout_seconds=12, max_retries=2)
        failing = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(TimeoutError("slow"))))
        with patch("agent0_protocol.responses_runtime.OpenAI", return_value=failing) as constructor:
            with self.assertRaisesRegex(CapabilityError, "TimeoutError"):
                ResponsesRuntime(config, self.registry)
            self.assertEqual(constructor.call_args.kwargs["timeout"], 12)
            self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)

    def test_multistep_call_ids_and_tool_snapshot(self):
        fake = FakeResponses()
        runtime = ResponsesRuntime(self.config, self.registry, client=SimpleNamespace(responses=fake), probe_on_init=False)
        trajectory = runtime.run([{"type": "message", "role": "user", "content": "use echo twice"}])
        self.assertEqual(trajectory.tools, self.registry.definitions())
        self.assertEqual([item["type"] for item in trajectory.items],
                         ["message", "function_call", "function_call_output", "function_call", "function_call_output", "message"])
        self.assertEqual([item["call_id"] for item in trajectory.items if item["type"] == "function_call"], ["run_1", "run_2"])
        for request in fake.requests[1:]:
            self.assertEqual(request["tools"], self.registry.definitions())
        self.assertTrue(verify_trajectory(trajectory, self.registry).valid)

    def test_reject_orphan_and_invalid_arguments(self):
        trajectory = CanonicalTrajectory("t", self.registry.definitions())
        trajectory.append({"type": "function_call_output", "call_id": "missing", "output": {}})
        with self.assertRaises(ProtocolError):
            trajectory.validate()
        trajectory.items = [{"type": "function_call", "call_id": "x", "name": "echo", "arguments": {"value": "bad"}}]
        with self.assertRaises(Exception):
            trajectory.validate()

    def test_tool_error_and_retry_fresh_id(self):
        failing = ToolRegistry()
        failing.register(self.registry.definitions()[0], lambda args, context: (_ for _ in ()).throw(RuntimeError("failed")))
        output = failing.execute({"type": "function_call", "name": "echo", "call_id": "old", "arguments": {"value": 1}})
        self.assertFalse(output["success"])
        trajectory = CanonicalTrajectory("retry", failing.definitions())
        trajectory.append({"type": "function_call", "name": "echo", "call_id": "old", "arguments": {"value": 1}})
        trajectory.append({"type": "function_call_output", "call_id": "old", "output": output})
        new_id = retry_function(trajectory, "old", failing)
        self.assertNotEqual(new_id, "old")

    def test_runtime_returns_structured_tool_error(self):
        failing = ToolRegistry()
        failing.register(self.registry.definitions()[0], lambda args, context: (_ for _ in ()).throw(RuntimeError("failed")))
        fake = FakeResponses()
        runtime = ResponsesRuntime(self.config, failing, client=SimpleNamespace(responses=fake), probe_on_init=False)
        trajectory = runtime.run([{"type": "message", "role": "user", "content": "echo twice"}])
        outputs = [item for item in trajectory.items if item["type"] == "function_call_output"]
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(item["output"]["success"] is False for item in outputs))
        self.assertFalse(verify_trajectory(trajectory, failing).valid)
        self.assertTrue(all(json.loads(item["output"])["success"] is False
                            for request in fake.requests[1:] for item in request["input"]
                            if isinstance(item, dict) and item.get("type") == "function_call_output"))

    def test_raw_rollout_requires_real_sample_logprobs(self):
        rollout = RawRollout([1], [1], [10, 11], [-0.1, None], [True, True], [True, True, True], [True, False], [10], policy_version="p1")
        rollout.validate()
        rollout.sampling_mask[1] = True
        with self.assertRaises(ProtocolError):
            rollout.validate()

    def test_qwen_model_adapter_only_token_boundary(self):
        tokenizer = FakeTokenizer()
        tokenizer.value = '<think>reason</think><tool_call>{"name":"echo","arguments":{"value":1}}</tool_call>'
        items = QwenModelAdapter(tokenizer).decode_items([100, 101])
        self.assertEqual([item["type"] for item in items], ["reasoning", "function_call"])
        self.assertIsInstance(items[1]["arguments"], dict)
        self.assertIn("<tool_call>", QwenModelAdapter(tokenizer).render(items, self.registry.definitions(), generate=False))

    def test_qwen_adapter_preserves_multiple_call_order_and_ids(self):
        tokenizer = FakeTokenizer()
        tokenizer.value = ('first<tool_call>{"name":"echo","arguments":{"value":1}}</tool_call>'
                           '<tool_call>{"name":"echo","arguments":{"value":2}}</tool_call>after')
        adapter = QwenModelAdapter(tokenizer)
        items = adapter.decode_items([1])
        self.assertEqual([item["type"] for item in items],
                         ["message", "function_call", "function_call", "message"])
        self.assertNotEqual(items[1]["call_id"], items[2]["call_id"])
        rendered = adapter.render([{"type": "function_call_output", "call_id": items[1]["call_id"],
                                    "output": {"success": True}}], self.registry.definitions(), generate=False)
        self.assertIn(items[1]["call_id"], rendered)

    def test_parse_verification_and_repair_json_robustness(self):
        markdown_verify = "```json\n{\"step_index\": 1, \"score\": 0.9, \"confidence\": 0.85, \"critique\": \"Good step\"}\n```"
        parsed_v = parse_verification_output(markdown_verify)
        self.assertIsNotNone(parsed_v)
        self.assertEqual(parsed_v["score"], 0.9)
        self.assertEqual(parsed_v["confidence"], 0.85)

        markdown_repair = "Thought before repair\n```json\n{\"action\": \"patch\", \"target_step\": 1, \"patch_type\": \"tool_call\"}\n```"
        parsed_r = parse_repair_instruction(markdown_repair)
        self.assertIsNotNone(parsed_r)
        self.assertEqual(parsed_r["action"], "PATCH")

    def test_runtime_run_verifier_and_repair(self):
        class VerifierResponses:
            def create(self, **request):
                return FakeResponse([{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": '{"step_index": 1, "score": 1.0, "confidence": 0.9, "critique": "OK"}'}
                ]}])

        runtime = ResponsesRuntime(self.config, self.registry, client=SimpleNamespace(responses=VerifierResponses()), probe_on_init=False)
        traj = CanonicalTrajectory("t1", self.registry.definitions())
        traj.append({"type": "message", "role": "user", "content": "1+1"})
        traj.append({"type": "message", "role": "assistant", "content": "2"})
        result = runtime.run_verifier(traj)
        self.assertEqual(result.get("score"), 1.0)
        self.assertEqual(result.get("confidence"), 0.9)

    def test_responses_adapter_handles_dict_and_string_arguments(self):
        adapter = ResponsesAdapter()
        # Case 1: arguments is a valid JSON string
        resp1 = FakeResponse([{"type": "function_call", "call_id": "c1", "name": "echo", "arguments": '{"value": 42}'}])
        items1 = adapter.output_items(resp1)
        self.assertEqual(items1[0]["arguments"], {"value": 42})

        # Case 2: arguments is already a dictionary
        resp2 = FakeResponse([{"type": "function_call", "call_id": "c2", "name": "echo", "arguments": {"value": 42}}])
        items2 = adapter.output_items(resp2)
        self.assertEqual(items2[0]["arguments"], {"value": 42})

    def test_qwen_adapter_compact_rendering_and_deduplication(self):
        tokenizer = FakeTokenizer()
        adapter = QwenModelAdapter(tokenizer)
        items = [{"type": "message", "role": "system", "content": "System directive."}]
        rendered = adapter.render(items, self.registry.definitions(), generate=False)
        self.assertIn("Available functions:", rendered)
        # Verify compact JSON separators (no ': ' spaces)
        self.assertIn('"name":"echo"', rendered)

        # Ensure no duplication if already rendered
        items_dup = [{"type": "message", "role": "system", "content": rendered}]
        rendered_dup = adapter.render(items_dup, self.registry.definitions(), generate=False)
        self.assertEqual(rendered_dup.count("Available functions:"), 1)

    def test_repair_inherits_image_context(self):
        class CaptureResponses:
            def __init__(self):
                self.requests = []

            def create(self, **request):
                self.requests.append(request)
                return FakeResponse([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Repaired"}]}])

        fake = CaptureResponses()
        runtime = ResponsesRuntime(self.config, self.registry, client=SimpleNamespace(responses=fake), probe_on_init=False)
        traj = CanonicalTrajectory("t_repair", self.registry.definitions())
        traj.append({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect"},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="},
            ],
        })
        traj.append({"type": "message", "role": "assistant", "content": "Need repair"})

        repaired_traj = runtime.run_repair(traj, {"issue": "check bbox"})
        self.assertEqual(repaired_traj.items[-1]["type"], "message")
        self.assertEqual(repaired_traj.items[-1]["content"][0]["text"], "Repaired")


if __name__ == "__main__":
    unittest.main()
