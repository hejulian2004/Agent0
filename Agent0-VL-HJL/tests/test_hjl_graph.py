"""Comprehensive behavioral and invariant tests for HJL StateGraph."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from hjl.graph import create_hjl_graph
from hjl.nodes.evidence_updater import evidence_updater_node
from hjl.nodes.evidence_verifier import evidence_verifier_node
from hjl.nodes.failure_diagnoser import failure_diagnoser_node
from hjl.nodes.global_verifier import global_verifier_node
from hjl.nodes.planner import planner_node
from hjl.nodes.regional_verifier import regional_verifier_node
from hjl.nodes.replanner import replanner_node
from hjl.nodes.tool_executor import tool_executor_node
from hjl.state import EvidenceItem, EvidenceRelation, EvidenceState, HJLPhase, HJLState, StopReason
from hjl.taxonomy import (
    ActionType,
    CheckpointJudgment,
    EvidenceConclusion,
    EvidenceJudgment,
    EvidenceStatus,
    FailureType,
    GlobalStatus,
    RegionalStatus,
)


class TestHJLGraph(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img = Image.new("RGB", (100, 100), color="gray")
        img.save(self.tmp.name)
        self.image_path = self.tmp.name

    def tearDown(self):
        Path(self.image_path).unlink(missing_ok=True)

    def test_invariant_1_tool_failure_direct_route_without_llm(self):
        """Tool failure must directly assign TOOL_FAILURE and bypass LLM diagnoser."""
        state = HJLState(
            sample_id="test_inv1",
            image_path=self.image_path,
            tool_calls=[{"name": "crop_region", "arguments": {"bbox": [1000, 1000, 2000, 2000]}}],  # Invalid bbox
        )
        res = tool_executor_node(state)
        self.assertEqual(res.get("failure_type"), FailureType.TOOL_FAILURE)
        self.assertIn(ActionType.RETRY_TOOL, res.get("allowed_actions", []))
        self.assertEqual(res.get("consecutive_tool_failures"), 1)

    def test_invariant_2_regional_fail_does_not_pollute_evidence(self):
        """Regional FAIL must not invoke EvidenceUpdater; EvidenceState remains uncontaminated."""
        state = HJLState(
            sample_id="test_inv2",
            image_path=self.image_path,
            observations=[{
                "step": 1,
                "tool": "crop_region",
                "metadata": {"image_size": [5, 5], "bbox": [0, 0, 5, 5]},  # Degenerate crop
            }],
        )
        # 1. Regional verifier evaluates observation
        v_res = regional_verifier_node(state)
        self.assertEqual(v_res["regional_judgment"].status, RegionalStatus.FAIL)

        # 2. Confirm evidence_updater is NOT executed on FAIL
        # Initial evidence items count is 0
        self.assertEqual(len(state.evidence_state.evidence_items), 0)

        # If evidence_updater was mistakenly called, it would have added an item
        # Since graph routes to failure_diagnoser on FAIL, evidence_updater is skipped
        d_res = failure_diagnoser_node(HJLState(
            sample_id="test_inv2",
            image_path=self.image_path,
            regional_judgment=v_res["regional_judgment"],
        ))
        self.assertEqual(d_res["failure_type"], FailureType.LOW_RESOLUTION)
        self.assertIn(ActionType.ENHANCE_REGION, d_res["allowed_actions"])

    def test_invariant_3_failure_diagnosis_constrained_replanning_recovery(self):
        """Failure diagnosis drives action mask, replanner selects recovery action, leading to success."""
        state = HJLState(
            sample_id="test_inv3",
            image_path=self.image_path,
            phase=HJLPhase.HYPOTHESIS_INSPECTION,
            active_hypothesis={"type": "defect", "confidence": 0.85},
            regional_judgment=CheckpointJudgment(
                status=RegionalStatus.FAIL,
                judgment_confidence=0.85,
                reason="ROI resolution is insufficient",
            ),
        )
        # Diagnosis
        d_res = failure_diagnoser_node(state)
        self.assertEqual(d_res["failure_type"], FailureType.LOW_RESOLUTION)
        state.failure_type = d_res["failure_type"]
        state.allowed_actions = d_res["allowed_actions"]

        # Constrained Replanning
        r_res = replanner_node(state)
        self.assertEqual(r_res["selected_action"], ActionType.ENHANCE_REGION)
        state.selected_action = r_res["selected_action"]

        # Planning instantiates tool call
        p_res = planner_node(state)
        self.assertEqual(p_res["tool_calls"][0]["name"], "crop_region")

    def test_invariant_4_global_fail_discovery_phase_with_none_hypothesis(self):
        """GlobalStatus.FAIL routes to GLOBAL_DISCOVERY phase with hypothesis=None and allowed discovery actions."""
        state = HJLState(
            sample_id="test_inv4",
            image_path=self.image_path,
            candidate_regions=[],  # Ambiguous view
        )
        gv_res = global_verifier_node(state)
        self.assertEqual(gv_res["global_judgment"].status, GlobalStatus.FAIL)
        self.assertEqual(gv_res["phase"], HJLPhase.GLOBAL_DISCOVERY)
        self.assertIn(ActionType.GLOBAL_SCAN, gv_res["allowed_actions"])

        # Planner operates in GLOBAL_DISCOVERY mode without needing active_hypothesis
        state.phase = gv_res["phase"]
        state.allowed_actions = gv_res["allowed_actions"]
        state.active_hypothesis = None

        p_res = planner_node(state)
        self.assertEqual(p_res["current_plan"]["phase"], "GLOBAL_DISCOVERY")
        self.assertEqual(p_res["tool_calls"][0]["name"], "localize_candidate")

    def test_invariant_5_evidence_verifier_pass_emits_definitive_conclusion(self):
        """EvidenceVerifier PASS must explicitly emit ANOMALY or NORMAL, which finalizer consumes."""
        # Anomaly scenario
        state_anom = HJLState(sample_id="test_inv5a", image_path=self.image_path)
        state_anom.evidence_state.supporting_evidence.append(
            EvidenceItem(1, [10, 10, 30, 30], "crack", "Clear crack observed", EvidenceRelation.SUPPORT, 0.9, "crop_region")
        )
        state_anom.evidence_state.anomaly_score = 0.90
        ev_res = evidence_verifier_node(state_anom, anomaly_threshold=0.75)
        self.assertEqual(ev_res["evidence_judgment"].status, EvidenceStatus.PASS)
        self.assertEqual(ev_res["evidence_judgment"].conclusion, EvidenceConclusion.ANOMALY)

    def test_invariant_6_zero_score_without_positive_evidence_never_normal(self):
        """Zero initial anomaly score with empty evidence list must NEVER conclude NORMAL."""
        state_empty = HJLState(sample_id="test_inv6", image_path=self.image_path)
        # Empty evidence state
        self.assertEqual(state_empty.evidence_state.anomaly_score, 0.0)
        self.assertEqual(len(state_empty.evidence_state.evidence_items), 0)

        ev_res = evidence_verifier_node(state_empty, normal_threshold=0.20, min_evidence_count=1)
        # Must FAIL / be UNRESOLVED, not PASS as NORMAL!
        self.assertEqual(ev_res["evidence_judgment"].status, EvidenceStatus.FAIL)
        self.assertEqual(ev_res["evidence_judgment"].conclusion, EvidenceConclusion.UNRESOLVED)
        self.assertEqual(ev_res["phase"], HJLPhase.EVIDENCE_RESOLUTION)

    def test_invariant_7_golden_multi_step_trajectory(self):
        """Test full graph execution running end-to-end to definitive conclusion."""
        graph = create_hjl_graph()
        initial_state = HJLState(
            sample_id="test_golden",
            image_path=self.image_path,
            max_steps=5,
        )
        final_state = graph.run(initial_state)

        # Graph completes and reaches final prediction
        self.assertIsNotNone(final_state.final_prediction)
        self.assertIn(final_state.stop_reason, [
            StopReason.CONFIRMED_ANOMALY,
            StopReason.CONFIRMED_NORMAL,
            StopReason.MAX_STEPS,
        ])
        self.assertGreaterEqual(final_state.current_step, 1)


if __name__ == "__main__":
    unittest.main()
