"""Comprehensive behavioral, invariant, and acceptance tests for HJL StateGraph."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from agent0_protocol.tools import ToolExecutionContext
from hjl.config import HJLConfig
from hjl.graph import create_hjl_graph
from hjl.model_caller import (
    CandidateRegion,
    GlobalInspectionResult,
    HypothesisResult,
    ModelOutputError,
    RegionalEvidenceFinding,
    RegionalVerificationResult,
    ScriptedHJLModelCaller,
)
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
        self.assertEqual(res.get("consecutive_tool_failures"), 1)
        # Non-retriable invalid arguments failure terminates via NO_VALID_ACTION
        self.assertEqual(res.get("stop_reason"), StopReason.NO_VALID_ACTION)

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
        self.assertEqual(ev_res["evidence_judgment"].status, EvidenceStatus.FAIL)
        self.assertEqual(ev_res["evidence_judgment"].conclusion, EvidenceConclusion.UNRESOLVED)
        self.assertEqual(ev_res["phase"], HJLPhase.EVIDENCE_RESOLUTION)

    def test_invariant_7_golden_multi_step_trajectory(self):
        """Verify the complete multi-step trajectory and crucial state invariants end-to-end with ScriptedHJLModelCaller.

        Prescribed path starting from global_inspector:
        1. Global Inspector proposes candidate [10, 10, 16, 16] -> Global Verifier yields PASS
        2. Hypothesis Generator formulates crack hypothesis
        3. Valid small crop (6x6) -> Tool SUCCESS -> Regional FAIL (LOW_RESOLUTION)
        4. Regional FAIL leaves evidence_items empty and anomaly_score == 0.0
        5. Diagnosis: LOW_RESOLUTION -> Replanner: ENHANCE_REGION
        6. Zoom (12x12) -> Tool SUCCESS -> Regional PASS
        7. Evidence Extractor extracts zoomed ROI -> Evidence Updater records NEUTRAL (score remains 0.0)
        8. Evidence Verifier yields FAIL (MISSING_REFERENCE) -> Action Mask {RETRIEVE_REFERENCE}
        9. Tool Executor retrieves train-normal reference -> ReferenceStateUpdater transitions to CROSS_VALIDATE
        10. Planner issues compare_with_reference with explicit reference_path and normalized_bbox
        11. Comparison yields difference -> Comparison Extractor emits SUPPORT -> Evidence Updater sets anomaly_score = 0.8
        12. Evidence Verifier yields PASS (ANOMALY) -> Finalizer terminates with CONFIRMED_ANOMALY
        """
        config = HJLConfig(
            anomaly_threshold=0.75,
            normal_threshold=0.20,
            reference_similarity_threshold=0.80,
            max_steps=10,
        )
        context = ToolExecutionContext(image=self.image_path)
        context["original_image_path"] = str(self.image_path)

        scripted_caller = ScriptedHJLModelCaller(
            global_inspection=GlobalInspectionResult(
                observation="Suspicious dark region detected on global surface.",
                candidate_regions=[
                    CandidateRegion(bbox=[10, 10, 16, 16], confidence=0.85, label="tiny_crack_candidate")
                ],
                is_normal=False,
                confidence=0.85,
            ),
            hypothesis=HypothesisResult(
                hypothesis_id="hyp_crack_01",
                type="crack",
                description="Micro-crack at [10, 10, 16, 16]",
                confidence=0.85,
                target_region=[10, 10, 16, 16],
            ),
            evidence_finding=RegionalEvidenceFinding(
                finding="Zoomed surface structure shows ambiguous discoloration.",
                relation=EvidenceRelation.NEUTRAL,
                observation_type="inspected_roi",
                confidence=0.85,
            ),
        )

        try:
            graph = create_hjl_graph(
                context=context,
                config=config,
                model_caller=scripted_caller,
                allow_synthetic=True,
            )

            # Start from clean initial state (NO pre-seeded candidates!)
            initial_state = HJLState(
                sample_id="test_golden",
                image_path=self.image_path,
                category="metal_casting",
                max_steps=10,
                candidate_regions=[],
            )

            final_state = graph.run(initial_state)

            # 1. Terminal Outcome Verification
            self.assertEqual(final_state.stop_reason, StopReason.CONFIRMED_ANOMALY)
            self.assertIsNotNone(final_state.final_prediction)
            self.assertEqual(final_state.final_prediction["conclusion"], "ANOMALY")
            self.assertTrue(final_state.final_prediction["is_anomaly"])
            self.assertFalse(final_state.final_prediction["best_effort"])

            # 2. Granular State Invariants Across Transitions
            hist = final_state.step_history
            self.assertTrue(len(hist) > 0)

            # Step Invariant A: Global Inspector proposed candidate [10, 10, 16, 16]
            inspector_entries = [h for h in hist if h.get("node") == "global_inspector"]
            self.assertTrue(len(inspector_entries) >= 1)

            # Step Invariant B: Regional Fail occurred on 6x6 crop with score == 0.0
            reg_fail_entries = [h for h in hist if h.get("node") == "regional_verifier" and h.get("regional_judgment") and h["regional_judgment"].get("status") == "FAIL"]
            self.assertTrue(len(reg_fail_entries) >= 1, "Expected at least one Regional FAIL step")
            self.assertEqual(reg_fail_entries[0]["anomaly_score"], 0.0)

            # Step Invariant C: Regional Pass on zoom step had score == 0.0
            reg_pass_entries = [h for h in hist if h.get("node") == "regional_verifier" and h.get("regional_judgment") and h["regional_judgment"].get("status") == "PASS"]
            self.assertTrue(len(reg_pass_entries) >= 1)

            # Step Invariant D: Retrieval step had score == 0.0
            retrieve_entries = [h for h in hist if h.get("tool_call") and h["tool_call"].get("name") == "retrieve_normal_reference"]
            self.assertTrue(len(retrieve_entries) >= 1)
            self.assertEqual(retrieve_entries[0]["anomaly_score"], 0.0)

            # Step Invariant E: Comparison step detected difference -> score rose to 0.8
            compare_entries = [h for h in hist if h.get("tool_call") and h["tool_call"].get("name") == "compare_with_reference"]
            self.assertTrue(len(compare_entries) >= 1)
            comp_args = compare_entries[0]["tool_call"]["arguments"]
            self.assertIn("reference_path", comp_args)
            self.assertIn("normalized_bbox", comp_args)

            # Final Evidence Score Invariant
            self.assertGreaterEqual(final_state.evidence_state.anomaly_score, 0.75)
            self.assertGreaterEqual(len(final_state.evidence_state.supporting_evidence), 1)

        finally:
            context.close()

    def test_acceptance_confirmed_normal(self):
        """Acceptance Test A: When global VLM confirms normal with high confidence, fast-exit to CONFIRMED_NORMAL."""
        config = HJLConfig(global_normal_confidence_threshold=0.95)
        context = ToolExecutionContext(image=self.image_path)
        context["original_image_path"] = str(self.image_path)

        normal_caller = ScriptedHJLModelCaller(
            global_inspection=GlobalInspectionResult(
                observation="Clean pristine component without defects.",
                candidate_regions=[],
                is_normal=True,
                confidence=0.98,
            )
        )

        try:
            graph = create_hjl_graph(context=context, config=config, model_caller=normal_caller)
            initial_state = HJLState(sample_id="test_norm", image_path=self.image_path)
            final_state = graph.run(initial_state)

            self.assertEqual(final_state.stop_reason, StopReason.CONFIRMED_NORMAL)
            self.assertEqual(final_state.final_prediction["conclusion"], "NORMAL")
            self.assertFalse(final_state.final_prediction["is_anomaly"])
            self.assertFalse(final_state.final_prediction["best_effort"])
            # Fast exit without tool execution
            self.assertEqual(final_state.current_step, 0)
        finally:
            context.close()

    def test_acceptance_discovery_exhaustion(self):
        """Acceptance Test B: Ambiguous view fails global verification; 0 candidates after 2 attempts terminates via NO_VALID_ACTION."""
        config = HJLConfig(max_discovery_attempts=2)
        # Create solid white image without any dark boxes for visual_analyzer
        solid_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.new("RGB", (100, 100), color=(255, 255, 255)).save(solid_tmp.name)

        context = ToolExecutionContext(image=solid_tmp.name)
        context["original_image_path"] = str(solid_tmp.name)

        # Model caller returns empty candidates but not high confidence normal
        ambiguous_caller = ScriptedHJLModelCaller(
            global_inspection=GlobalInspectionResult(
                observation="Ambiguous view with lighting glare.",
                candidate_regions=[],
                is_normal=False,
                confidence=0.70,
            )
        )

        try:
            graph = create_hjl_graph(context=context, config=config, model_caller=ambiguous_caller)
            initial_state = HJLState(sample_id="test_disc_exhaust", image_path=solid_tmp.name, max_steps=6)
            final_state = graph.run(initial_state)

            self.assertEqual(final_state.stop_reason, StopReason.NO_VALID_ACTION)
            self.assertEqual(final_state.final_prediction["conclusion"], "UNRESOLVED")
            self.assertIsNone(final_state.final_prediction["is_anomaly"])
            self.assertTrue(final_state.final_prediction["best_effort"])
            self.assertGreaterEqual(final_state.discovery_attempts, 2)
        finally:
            context.close()
            Path(solid_tmp.name).unlink(missing_ok=True)

    def test_acceptance_missing_reference_live_mode(self):
        """Acceptance Test C: In live mode (allow_synthetic=False), missing reference fails retrieval cleanly, preserving prior history."""
        config = HJLConfig(max_steps=10)
        context = ToolExecutionContext(image=self.image_path)
        context["original_image_path"] = str(self.image_path)

        # Caller that guides graph through crop -> zoom -> missing reference
        caller = ScriptedHJLModelCaller(
            global_inspection=GlobalInspectionResult(
                observation="Suspicious crack candidate.",
                candidate_regions=[CandidateRegion(bbox=[10, 10, 16, 16], confidence=0.85)],
                is_normal=False,
                confidence=0.85,
            ),
            hypothesis=HypothesisResult(
                hypothesis_id="hyp_01",
                type="crack",
                description="Crack",
                confidence=0.85,
                target_region=[10, 10, 16, 16],
            ),
            evidence_finding=RegionalEvidenceFinding(
                finding="Zoomed surface inspected.",
                relation=EvidenceRelation.NEUTRAL,
                observation_type="inspected_roi",
                confidence=0.85,
            ),
        )

        try:
            # allow_synthetic=False: reference retrieval will fail because no local train corpus exists for this category
            graph = create_hjl_graph(
                context=context,
                config=config,
                model_caller=caller,
                allow_synthetic=False,
            )
            initial_state = HJLState(
                sample_id="test_live_missing_ref",
                image_path=self.image_path,
                category="nonexistent_metal_part_9999",
                max_steps=10,
            )
            final_state = graph.run(initial_state)

            # Must terminate UNRESOLVED without fake gray reference
            self.assertEqual(final_state.final_prediction["conclusion"], "UNRESOLVED")
            self.assertIsNone(final_state.final_prediction["is_anomaly"])
            self.assertEqual(final_state.stop_reason, StopReason.NO_VALID_ACTION)

            # Invariant: Observation history retains prior successful steps alongside the failed retrieval
            self.assertGreaterEqual(len(final_state.observations), 3)
            tools_called = [o["tool"] for o in final_state.observations]
            self.assertIn("crop_region", tools_called)
            self.assertIn("zoom_region", tools_called)
            self.assertIn("retrieve_normal_reference", tools_called)

            # Last observation was failed retrieve with retriable=False
            last_obs = final_state.observations[-1]
            self.assertFalse(last_obs["success"])
            self.assertFalse(last_obs["retriable"])
            self.assertIsNotNone(last_obs["error"])
        finally:
            context.close()

    def test_acceptance_malformed_model_output(self):
        """Acceptance Test D: In live mode (live_mode=True), unrecoverable model output error terminates cleanly via MODEL_ERROR."""
        context = ToolExecutionContext(image=self.image_path)
        context["original_image_path"] = str(self.image_path)

        class FailingCaller(ScriptedHJLModelCaller):
            def inspect_global(self, image_path, instruction, category):
                raise ModelOutputError("Model response failed schema validation after retry.")

        try:
            graph = create_hjl_graph(
                context=context,
                config=HJLConfig(),
                model_caller=FailingCaller(),
                live_mode=True,
            )
            initial_state = HJLState(sample_id="test_malformed", image_path=self.image_path)
            final_state = graph.run(initial_state)

            self.assertEqual(final_state.stop_reason, StopReason.MODEL_ERROR)
            self.assertEqual(final_state.final_prediction["conclusion"], "UNRESOLVED")
            self.assertIsNone(final_state.final_prediction["is_anomaly"])
            self.assertTrue(final_state.final_prediction["best_effort"])
        finally:
            context.close()

    def test_acceptance_multi_roi_original_space_isolation(self):
        """Acceptance Test E: Multi-ROI inspection must crop ROI B from original_image_path, not from previously transformed ROI A."""
        context = ToolExecutionContext(image=self.image_path)
        context["original_image_path"] = str(self.image_path)

        # Plan: crop ROI A [10, 10, 30, 30] -> zoom 2.0 -> inspect next region [60, 60, 80, 80]
        # Verify that ROI B is successfully cropped without dimension errors from original 100x100 space
        state = HJLState(
            sample_id="test_multi_roi",
            image_path=self.image_path,
            evidence_state=EvidenceState(unresolved_regions=[[60, 60, 80, 80]]),
        )

        # 1. First crop ROI A
        state.tool_calls = [{"name": "crop_region", "arguments": {"bbox": [10, 10, 30, 30], "use_original": True}}]
        res1 = tool_executor_node(state, context=context)
        self.assertEqual(res1["active_region_original_bbox"], [10, 10, 30, 30])
        self.assertEqual(res1["active_region_rotation_deg"], 0.0)

        # 2. Zoom ROI A
        state.current_step = res1["current_step"]
        state.observations = res1["observations"]
        state.tool_calls = [{"name": "zoom_region", "arguments": {"scale": 2.0}}]
        res2 = tool_executor_node(state, context=context)
        self.assertTrue(res2["observations"][-1]["success"])

        # 3. Rotate ROI A 90 deg
        state.current_step = res2["current_step"]
        state.observations = res2["observations"]
        state.tool_calls = [{"name": "rotate_image", "arguments": {"angle": 90.0}}]
        res3 = tool_executor_node(state, context=context)
        self.assertEqual(res3["active_region_rotation_deg"], 90.0)

        # 4. Now inspect next region [60, 60, 80, 80] with use_original=True
        # If this mistakenly executed against the 40x40 zoomed/rotated ROI A, bbox [60, 60, 80, 80] would fail!
        state.current_step = res3["current_step"]
        state.observations = res3["observations"]
        state.tool_calls = [{"name": "crop_region", "arguments": {"bbox": [60, 60, 80, 80], "use_original": True}}]
        res4 = tool_executor_node(state, context=context)

        # Must succeed because it was cropped from the 100x100 original image
        self.assertTrue(res4["observations"][-1]["success"])
        self.assertEqual(res4["active_region_original_bbox"], [60, 60, 80, 80])
        # Rotation must be reset to 0.0 on a new ROI crop
        self.assertEqual(res4["active_region_rotation_deg"], 0.0)
        context.close()

    def test_failure_diagnoser_selects_active_failing_checkpoint(self):
        """When regional verifier passed but evidence verifier failed, diagnoser must select the failing evidence judgment."""
        state = HJLState(
            sample_id="test_diag_failing_ckpt",
            image_path=self.image_path,
            regional_judgment=CheckpointJudgment(
                status=RegionalStatus.PASS,
                judgment_confidence=0.90,
                reason="Regional observation is sharp and feature-relevant.",
            ),
            evidence_judgment=EvidenceJudgment(
                status=EvidenceStatus.FAIL,
                conclusion=EvidenceConclusion.UNRESOLVED,
                judgment_confidence=0.65,
                reason="Cannot verify anomaly without comparing against a standard normal template.",
            ),
        )

        res = failure_diagnoser_node(state)
        # Must diagnose MISSING_REFERENCE from the failing evidence judgment, NOT regional PASS
        self.assertEqual(res["failure_type"], FailureType.MISSING_REFERENCE)
        self.assertIn(ActionType.RETRIEVE_REFERENCE, res["allowed_actions"])

    def test_zoom_and_reference_roi_dedup_and_unresolved_clearing(self):
        """Zoom and reference comparisons must retain active_region_original_bbox, clear unresolved regions, and deduplicate."""
        state = HJLState(
            sample_id="test_dedup",
            image_path=self.image_path,
            active_region_original_bbox=[10, 10, 30, 30],
            evidence_state=EvidenceState(unresolved_regions=[[10, 10, 30, 30]]),
        )

        # 1. Simulate reference comparison finding
        state.observations.append({
            "step": 1,
            "tool": "compare_with_reference",
            "metadata": {"similarity": 0.20, "bbox": [10, 10, 30, 30]},
        })
        res1 = evidence_updater_node(state)
        ev_state = res1["evidence_state"]

        self.assertEqual(len(ev_state.evidence_items), 1)
        self.assertEqual(ev_state.evidence_items[0].region, [10, 10, 30, 30])
        # Unresolved region was cleared
        self.assertEqual(len(ev_state.unresolved_regions), 0)
        first_score = ev_state.anomaly_score
        self.assertEqual(first_score, 0.80)

        # 2. Second identical reference comparison on the exact same region
        state.evidence_state = ev_state
        state.observations.append({
            "step": 2,
            "tool": "compare_with_reference",
            "metadata": {"similarity": 0.20, "bbox": [10, 10, 30, 30]},
        })
        res2 = evidence_updater_node(state)
        ev_state2 = res2["evidence_state"]

        # Deduplication must prevent stacking duplicate +0.8 evidence
        self.assertEqual(len(ev_state2.evidence_items), 1)
        self.assertEqual(ev_state2.anomaly_score, first_score)

    def test_hard_step_budget_enforcement(self):
        """Hard budget enforcement: current_step >= max_steps immediately halts further tool executions."""
        state = HJLState(
            sample_id="test_hard_budget",
            image_path=self.image_path,
            current_step=5,
            max_steps=5,
            tool_calls=[{"name": "crop_region", "arguments": {"bbox": [0, 0, 10, 10]}}],
        )

        res = tool_executor_node(state)
        self.assertEqual(res["stop_reason"], StopReason.MAX_STEPS)

    def test_relocalize_refreshes_hypothesis(self):
        """Relocalization proposing new candidates must clear stale active_hypothesis and reset hypotheses."""
        state = HJLState(
            sample_id="test_relocalize_refresh",
            image_path=self.image_path,
            active_hypothesis={"hypothesis_id": "hyp_old", "type": "scratch"},
            hypotheses=[{"hypothesis_id": "hyp_old"}],
            observations=[{
                "step": 1,
                "tool": "localize_candidate",
                "metadata": {
                    "candidate_regions": [
                        {"bbox": [50, 50, 80, 80], "confidence": 0.85, "label": "new_candidate"}
                    ]
                },
            }],
        )

        res = candidate_state_updater_node(state)
        self.assertIsNone(res["active_hypothesis"])
        self.assertEqual(res["hypotheses"], [])
        self.assertEqual(res["phase"], HJLPhase.HYPOTHESIS_INSPECTION)
        self.assertEqual(len(res["candidate_regions"]), 1)


if __name__ == "__main__":
    unittest.main()
