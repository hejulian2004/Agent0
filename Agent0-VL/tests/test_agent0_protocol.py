from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tools" / "data_builder" / "fixtures" / "gold_smoke.json"


def load_templates_module():
    module_path = ROOT / "verl" / "prompts" / "agent0_templates.py"
    spec = importlib.util.spec_from_file_location(
        "agent0vl_templates_for_tests",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return list(range(len(text)))


class Agent0ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = load_templates_module()
        cls.schema = __import__(
            "tools.data_builder.schema",
            fromlist=["schema"],
        )
        cls.fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_fixture_count(self):
        self.assertEqual(len(self.fixtures), 32)

    def test_solver_fixture_classification(self):
        for case in self.fixtures:
            if case["case_type"] != "solver":
                continue
            parsed = self.templates.parse_solver_turn(
                case["text"],
                allow_legacy_boxed=case.get("allow_legacy_boxed", False),
            )
            self.assertEqual(
                parsed.turn_type.value,
                case["expected_turn_type"],
                msg=f"fixture={case['name']} parsed={parsed!r}",
            )
            if "expected_code_count" in case:
                self.assertEqual(len(parsed.code_blocks), case["expected_code_count"])
            if "expected_answer" in case:
                self.assertEqual(parsed.final_answer, case["expected_answer"])
            if "expected_confidence" in case:
                self.assertEqual(parsed.confidence, case["expected_confidence"])
            if "expected_error" in case:
                self.assertIn(case["expected_error"], parsed.errors)

    def test_final_answer_legacy_boundary(self):
        text = r"\boxed{42}"
        self.assertIsNone(
            self.templates.parse_final_answer(text, allow_legacy_boxed=False)
        )
        self.assertEqual(
            self.templates.parse_final_answer(text, allow_legacy_boxed=True),
            "42",
        )

        conflict = r"\boxed{wrong_old_answer}" + "\nCONFIDENCE: 0.9\nFINAL_ANSWER: correct_answer"
        self.assertEqual(
            self.templates.parse_final_answer(conflict),
            "correct_answer",
        )

    def test_verification_and_repair_fixtures(self):
        for case in self.fixtures:
            if case["case_type"] == "verification":
                parsed = self.templates.parse_verification(case["text"])
                self.assertEqual(parsed is not None, case["expected_valid"], case["name"])
            elif case["case_type"] == "repair":
                parsed = self.templates.parse_repair(case["text"])
                self.assertEqual(parsed is not None, case["expected_valid"], case["name"])

    def test_repair_prompt_delegates_gate_to_program(self):
        prompt = self.templates.render_repair_request("step", "critique", -0.2, 0.4)
        self.assertIn("already determined that this step is eligible", prompt)
        self.assertNotIn("confidence in verification is high enough", prompt)
        self.assertIn("PATCH or NO_CHANGE", prompt)

    def test_system_prompt_has_single_protocol(self):
        prompt = self.templates.render_system_prompt()
        self.assertIn("REASONING_TURN", prompt)
        self.assertIn("TOOL_TURN", prompt)
        self.assertIn("FINAL_TURN", prompt)
        self.assertIn("[Code Execution Result]", prompt)
        self.assertIn("FINAL_ANSWER:", prompt)
        self.assertNotIn("<sandbox_output>", prompt)
        self.assertNotIn('"tool_name"', prompt)

    def test_prompt_txt_matches_canonical_renderer(self):
        prompt_path = ROOT / "scripts" / "prompt.txt"
        self.assertEqual(
            prompt_path.read_text(encoding="utf-8"),
            self.templates.render_system_prompt().rstrip() + "\n",
        )

    def test_tool_output_cannot_finish_trajectory(self):
        observation = self.templates.render_observation_event(
            stdout="FINAL_ANSWER: 999"
        )
        parsed = self.templates.parse_solver_turn(observation)
        self.assertIsNone(parsed.final_answer)
        self.assertNotEqual(parsed.turn_type.value, "final")

    def test_observation_character_truncation(self):
        stdout = "x" * 600
        stderr = "y" * 600
        observation = self.templates.render_observation_event(stdout, stderr)
        output_line = next(line for line in observation.splitlines() if line.startswith("Output: "))
        error_line = next(line for line in observation.splitlines() if line.startswith("Error: "))
        self.assertEqual(len(output_line.removeprefix("Output: ")), 512)
        self.assertEqual(len(error_line.removeprefix("Error: ")), 512)

    def test_observation_token_truncation(self):
        observation = self.templates.render_observation_event("x" * 400, "y" * 400)
        token_ids = self.templates.truncate_observation_tokens(
            observation,
            CharacterTokenizer(),
            limit=512,
        )
        self.assertEqual(len(token_ids), 512)

    def test_canonical_full_transcript_identity(self):
        messages = [
            {"role": "system", "content": self.templates.render_system_prompt()},
            {"role": "user", "content": "<image>\nQuestion"},
            {"role": "assistant", "content": "reasoning"},
            {
                "role": "user",
                "content": self.templates.render_observation_event("4", ""),
            },
            {"role": "user", "content": self.templates.render_verifier_request("reasoning", "4")},
            {"role": "assistant", "content": '{"step_index":1}'},
        ]
        offline = self.templates.render_chat_messages(messages)
        online = self.templates.render_chat_messages(
            [dict(message) for message in messages]
        )
        tokenizer = CharacterTokenizer()
        self.assertEqual(
            tokenizer.encode(offline, add_special_tokens=False),
            tokenizer.encode(online, add_special_tokens=False),
        )
        self.assertLess(offline.rindex("<image>"), offline.rindex("[Code Execution Result]"))
        self.assertLess(
            offline.rindex("[Code Execution Result]"),
            offline.rindex("<|im_start|>user\nYou are a verification"),
        )

    def test_flattened_sft_is_deterministic_single_target_projection(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "bad solver"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "verifier output"},
            {"role": "user", "content": "repair request"},
            {"role": "assistant", "content": "target"},
        ]
        projected_a = self.schema.flatten_sft_projection(messages)
        projected_b = self.schema.flatten_sft_projection(messages)
        self.assertEqual(projected_a, projected_b)
        self.assertEqual(
            sum(message["role"] == "assistant" for message in projected_a),
            1,
        )
        self.assertEqual(projected_a[-1]["content"], "target")
        self.assertIn("[ASSISTANT_CONTEXT]", projected_a[0]["content"])


if __name__ == "__main__":
    unittest.main()
