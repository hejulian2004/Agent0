"""Unit tests for HJL trajectory logging and CanonicalTrajectory export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent0_protocol.schema import CanonicalTrajectory, SCHEMA_VERSION
from hjl.state import EvidenceItem, EvidenceRelation, HJLPhase, HJLState, StopReason
from hjl.trajectory import append_trajectory_step, to_canonical_trajectory


class TestHJLTrajectory(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.traj_file = Path(self.tmp_dir.name) / "test_traj.jsonl"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_jsonl_append_and_read(self):
        record = {
            "sample_id": "s_123",
            "step": 2,
            "phase": HJLPhase.HYPOTHESIS_INSPECTION.value,
            "allowed_actions": ["ENHANCE_REGION"],
            "selected_action": "ENHANCE_REGION",
            "anomaly_score": 0.65,
            "judgment_confidence": 0.88,
            "diagnosis_confidence": 0.90,
            "stop_reason": None,
            "stop": False,
        }
        append_trajectory_step(self.traj_file, record)

        lines = self.traj_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["sample_id"], "s_123")
        self.assertEqual(data["selected_action"], "ENHANCE_REGION")
        self.assertEqual(data["anomaly_score"], 0.65)

    def test_to_canonical_trajectory_schema_conformance(self):
        state = HJLState(
            sample_id="s_canonical_test",
            image_path="/tmp/fake.png",
            stop_reason=StopReason.CONFIRMED_ANOMALY,
        )
        state.observations.append({
            "step": 1,
            "tool": "crop_region",
            "arguments": {"bbox": [0, 0, 30, 30], "use_original": True},
            "success": True,
            "output_path": "/tmp/crop.png",
            "metadata": {"image_size": [30, 30]},
            "error": None,
            "retriable": False,
        })
        state.evidence_state.supporting_evidence.append(
            EvidenceItem(1, [0, 0, 30, 30], "defect", "Crack visible", EvidenceRelation.SUPPORT, 0.9, "crop_region")
        )
        state.evidence_state.anomaly_score = 0.90

        traj = to_canonical_trajectory(state)

        # Validate trajectory adheres to agent0.responses.v1
        self.assertIsInstance(traj, CanonicalTrajectory)
        self.assertEqual(traj.schema_version, SCHEMA_VERSION)
        traj.validate()

        # Check initial user message contains both input_text and input_image
        user_msg = traj.items[0]
        self.assertEqual(user_msg["type"], "message")
        self.assertEqual(user_msg["role"], "user")
        content_types = [p["type"] for p in user_msg["content"]]
        self.assertIn("input_text", content_types)
        self.assertIn("input_image", content_types)
        img_part = next(p for p in user_msg["content"] if p["type"] == "input_image")
        self.assertEqual(img_part["image_url"], "/tmp/fake.png")

        # Check call-output pairing and faithful tool naming
        calls = [it for it in traj.items if it.get("type") == "function_call"]
        outputs = [it for it in traj.items if it.get("type") == "function_call_output"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(calls[0]["call_id"], outputs[0]["call_id"])
        self.assertEqual(calls[0]["name"], "crop_region")
        self.assertEqual(calls[0]["arguments"], {"bbox": [0, 0, 30, 30], "use_original": True})
        self.assertTrue(outputs[0]["output"]["success"])

    def test_to_canonical_trajectory_faithful_failures(self):
        """Tool failure in observation must NOT be rewritten as success=True in canonical export."""
        state = HJLState(
            sample_id="s_canonical_fail",
            image_path="/tmp/fake.png",
            stop_reason=StopReason.NO_VALID_ACTION,
        )
        state.observations.append({
            "step": 1,
            "tool": "retrieve_normal_reference",
            "arguments": {"category": "metal_casting", "allow_synthetic": False, "corpus_dir": "/tmp/corpus"},
            "success": False,
            "output_path": None,
            "metadata": {},
            "error": "No train-normal reference available in reference corpus",
            "retriable": False,
        })

        traj = to_canonical_trajectory(state)
        traj.validate()

        calls = [it for it in traj.items if it.get("type") == "function_call"]
        outputs = [it for it in traj.items if it.get("type") == "function_call_output"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "retrieve_normal_reference")
        # In model-visible canonical trajectory, runtime arguments (category, allow_synthetic, corpus_dir) are stripped
        self.assertEqual(calls[0]["arguments"], {})

        # Verify state.observations was NOT mutated (retains full runtime audit arguments)
        self.assertEqual(
            state.observations[0]["arguments"],
            {"category": "metal_casting", "allow_synthetic": False, "corpus_dir": "/tmp/corpus"},
        )

        # Must faithfully record success=False and error
        self.assertFalse(outputs[0]["output"]["success"])
        self.assertEqual(outputs[0]["output"]["error"], "No train-normal reference available in reference corpus")
        self.assertFalse(outputs[0]["output"]["retriable"])

    def test_to_canonical_trajectory_agent_visible_schemas_and_isolation(self):
        """Canonical trajectory must use agent-visible tool schemas and isolate runtime arguments."""
        state = HJLState(
            sample_id="s_canonical_iso",
            image_path="/tmp/fake.png",
            stop_reason=StopReason.CONFIRMED_NORMAL,
        )
        state.observations.extend([
            {
                "step": 1,
                "tool": "retrieve_normal_reference",
                "arguments": {"category": "bottle", "allow_synthetic": False, "corpus_dir": "/data/corpus"},
                "success": True,
                "output_path": "/tmp/ref.png",
                "metadata": {"split": "train", "is_normal": True},
                "error": None,
                "retriable": False,
            },
            {
                "step": 2,
                "tool": "crop_region",
                "arguments": {"bbox": [5, 5, 25, 25], "use_original": True},
                "success": True,
                "output_path": "/tmp/crop.png",
                "metadata": {"image_size": [20, 20]},
                "error": None,
                "retriable": False,
            },
        ])

        traj = to_canonical_trajectory(state)
        traj.validate()

        # Verify tool schemas in canonical trajectory are agent-visible
        ret_tool = next(t for t in traj.tools if t["name"] == "retrieve_normal_reference")
        props = ret_tool["parameters"]["properties"]
        self.assertEqual(props, {})
        self.assertEqual(ret_tool["parameters"]["required"], [])

        calls = [it for it in traj.items if it.get("type") == "function_call"]
        self.assertEqual(len(calls), 2)
        # retrieve_normal_reference function_call arguments stripped of runtime-only parameters
        self.assertEqual(calls[0]["arguments"], {})
        # crop_region retains model-semantic parameters
        self.assertEqual(calls[1]["arguments"], {"bbox": [5, 5, 25, 25], "use_original": True})

        # Verify state.observations still holds complete runtime execution arguments
        self.assertEqual(
            state.observations[0]["arguments"],
            {"category": "bottle", "allow_synthetic": False, "corpus_dir": "/data/corpus"},
        )


if __name__ == "__main__":
    unittest.main()
