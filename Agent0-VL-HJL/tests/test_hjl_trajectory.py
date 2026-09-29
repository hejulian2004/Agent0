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
            "arguments": {"bbox": [0, 0, 30, 30]},
            "metadata": {"image_size": [30, 30]},
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

        # Check call-output pairing
        calls = [it for it in traj.items if it.get("type") == "function_call"]
        outputs = [it for it in traj.items if it.get("type") == "function_call_output"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(calls[0]["call_id"], outputs[0]["call_id"])


if __name__ == "__main__":
    unittest.main()
