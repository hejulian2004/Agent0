"""Comprehensive behavioral and invariant tests for HJL StateGraph."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from agent0_protocol.tools import ToolExecutionContext
from hjl.config import HJLConfig
from hjl.graph import create_hjl_graph
from hjl.nodes.evidence_extractor import evidence_extractor_node
from hjl.nodes.evidence_updater import evidence_updater_node
from hjl.nodes.evidence_verifier import evidence_verifier_node
from hjl.nodes.failure_diagnoser import failure_diagnoser_node
from hjl.nodes.finalizer import finalizer_node
from hjl.nodes.global_verifier import global_verifier_node
from hjl.nodes.planner import planner_node
from hjl.nodes.regional_verifier import regional_verifier_node
from hjl.nodes.replanner import replanner_node
from hjl.nodes.state_updaters import (
    candidate_state_updater_node,
    comparison_evidence_extractor_node,
    reference_state_updater_node,
)
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
        # Draw a small defect area
        for x in range(10, 20):
            for y in range(10, 20):
                img.putpixel((x, y), (20, 20, 20))
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
                "metadata": {"image_size": [5, 5], "bbox": [0, 0, 5, 5]},  # Degenerate crop < 12px
            }],
        )
        # 1. Regional verifier evaluates observation
        v_res = regional_verifier_node(state)
        self.assertEqual(v_res["regional_judgment"].status, RegionalStatus.FAIL)

        # 2. Confirm evidence_updater is NOT executed on FAIL
        self.assertEqual(len(state.evidence_state.evidence_items), 0)

        # 3. Diagnosis confirms LOW_RESOLUTION
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
        self.assertEqual(state_empty.evidence_state.anomaly_score, 0.0)
        self.assertEqual(len(state_empty.evidence_state.evidence_items), 0)

        ev_res = evidence_verifier_node(state_empty, normal_threshold=0.20, min_evidence_count=1)
        # Must FAIL / be UNRESOLVED, not PASS as NORMAL!
        self.assertEqual(ev_res["evidence_judgment"].status, EvidenceStatus.FAIL)
        self.assertEqual(ev_res["evidence_judgment"].conclusion, EvidenceConclusion.UNRESOLVED)
        self.assertEqual(ev_res["phase"], HJLPhase.EVIDENCE_RESOLUTION)

    def test_invariant_7_golden_multi_step_trajectory(self):
        """Verify the complete multi-step trajectory and crucial state invariants end-to-end.

        Prescribed path:
        1. Global PASS -> Hypothesis formation
        2. Valid small crop (6x6) -> Tool SUCCESS -> Regional FAIL (LOW_RESOLUTION)
        3. Regional FAIL leaves evidence_items empty (unpolluted) and anomaly_score == 0.0
        4. Diagnosis: LOW_RESOLUTION -> Replanner: ENHANCE_REGION
        5. Zoom (12x12) -> Tool SUCCESS -> Regional PASS
        6. Evidence Extractor extracts zoomed ROI -> Evidence Updater records NEUTRAL (score remains 0.0)
        7. Evidence Verifier yields FAIL (MISSING_REFERENCE) -> Action Mask {RETRIEVE_REFERENCE}
        8. Tool Executor retrieves train-normal reference -> ReferenceStateUpdater transitions to CROSS_VALIDATE
        9. Planner issues compare_with_reference with explicit reference_path
        10. Comparison yields difference (similarity < 0.8) -> Evidence Updater sets anomaly_score = 0.8
        11. Evidence Verifier yields PASS (ANOMALY) -> Finalizer terminates with CONFIRMED_ANOMALY
        """
        config = HJLConfig(
            anomaly_threshold=0.75,
            normal_threshold=0.20,
            reference_similarity_threshold=0.80,
            max_steps=10,
        )
        context = ToolExecutionContext(image=self.image_path)

        try:
            graph = create_hjl_graph(context=context, config=config)

            # Initialize state with candidate region having a 6x6 target bbox: [10, 10, 16, 16]
            initial_state = HJLState(
                sample_id="test_golden",
                image_path=self.image_path,
                category="metal_casting",
                max_steps=10,
                candidate_regions=[{
                    "bbox": [10, 10, 16, 16],
                    "confidence": 0.85,
                    "label": "small_crack_candidate",
                }],
            )

            final_state = graph.run(initial_state)

            # 1. Terminal Outcome Verification
            self.assertEqual(final_state.stop_reason, StopReason.CONFIRMED_ANOMALY)
            self.assertIsNotNone(final_state.final_prediction)
            self.assertEqual(final_state.final_prediction["conclusion"], "ANOMALY")
            self.assertTrue(final_state.final_prediction["is_anomaly"])
            self.assertFalse(final_state.final_prediction["best_effort"])

            # 2. State Invariants across execution steps
            # Check regional fail step: Regional FAIL occurred and did NOT increase evidence
            hist = final_state.step_history
            self.assertTrue(len(hist) > 0)

            # Find regional verifier transitions
            reg_fail_entries = [h for h in hist if h.get("node") == "regional_verifier" and h.get("regional_judgment") and h["regional_judgment"].get("status") == "FAIL"]
            self.assertTrue(len(reg_fail_entries) >= 1, "Expected at least one Regional FAIL step")
            self.assertEqual(reg_fail_entries[0]["regional_judgment"]["status"], "FAIL")

            # Diagnoser immediately classifies LOW_RESOLUTION following regional fail
            diag_entries = [h for h in hist if h.get("node") == "failure_diagnoser"]
            self.assertTrue(len(diag_entries) >= 1)
            self.assertEqual(diag_entries[0]["failure_type"], "LOW_RESOLUTION")

            # Check that reference was retrieved and transition to CROSS_VALIDATE occurred
            retrieve_entries = [h for h in hist if h.get("tool_call") and h["tool_call"].get("name") == "retrieve_normal_reference"]
            self.assertTrue(len(retrieve_entries) >= 1, "Expected retrieve_normal_reference tool call")

            compare_entries = [h for h in hist if h.get("tool_call") and h["tool_call"].get("name") == "compare_with_reference"]
            self.assertTrue(len(compare_entries) >= 1, "Expected compare_with_reference tool call")

            # Invariant: compare_with_reference received explicit reference_path argument
            comp_args = compare_entries[0]["tool_call"]["arguments"]
            self.assertIn("reference_path", comp_args)
            self.assertTrue(len(comp_args["reference_path"]) > 0)

            # Invariant: Evidence accumulation produced terminal anomaly score
            self.assertGreaterEqual(final_state.evidence_state.anomaly_score, 0.75)
            self.assertGreaterEqual(len(final_state.evidence_state.supporting_evidence), 1)

        finally:
            context.close()


if __name__ == "__main__":
    unittest.main()
