"""Unit tests for typed model caller schemas and invariants."""

from __future__ import annotations

import unittest

from hjl.model_caller import (
    CandidateRegion,
    EvidenceVerificationResult,
    MockHJLModelCaller,
    ReactAction,
    ReactDecisionResult,
)
from hjl.state import EvidenceState
from hjl.taxonomy import EvidenceConclusion, EvidenceStatus


class TestHJLModelCaller(unittest.TestCase):
    def test_candidate_region_geometry_and_bounds_validation(self):
        # Valid candidate
        cand = CandidateRegion(bbox=[10, 20, 50, 60], confidence=0.90)
        self.assertEqual(cand.bbox, [10, 20, 50, 60])
        cand.validate_bounds(100, 100)  # Should pass

        # Invalid geometry: x1 >= x2
        with self.assertRaises(ValueError):
            CandidateRegion(bbox=[50, 20, 10, 60], confidence=0.90)

        # Invalid geometry: length != 4
        with self.assertRaises(ValueError):
            CandidateRegion(bbox=[10, 20, 50], confidence=0.90)

        # Confidence out of [0, 1]
        with self.assertRaises(ValueError):
            CandidateRegion(bbox=[10, 20, 50, 60], confidence=1.5)

        # Exceeds image bounds
        with self.assertRaises(ValueError):
            cand.validate_bounds(40, 40)

    def test_evidence_verification_result_cross_field_constraints(self):
        # PASS with ANOMALY -> valid
        res_anom = EvidenceVerificationResult(
            status=EvidenceStatus.PASS,
            conclusion=EvidenceConclusion.ANOMALY,
            judgment_confidence=0.95,
            reason="Confirmed anomaly",
        )
        self.assertEqual(res_anom.status, EvidenceStatus.PASS)

        # PASS with UNRESOLVED -> forbidden by invariant!
        with self.assertRaises(ValueError):
            EvidenceVerificationResult(
                status=EvidenceStatus.PASS,
                conclusion=EvidenceConclusion.UNRESOLVED,
                judgment_confidence=0.95,
                reason="Invalid state",
            )

        # FAIL with ANOMALY -> forbidden by invariant!
        with self.assertRaises(ValueError):
            EvidenceVerificationResult(
                status=EvidenceStatus.FAIL,
                conclusion=EvidenceConclusion.ANOMALY,
                judgment_confidence=0.60,
                reason="Invalid state",
            )

    def test_react_decision_result_action_constraints(self):
        # Valid TOOL_CALL
        call_dec = ReactDecisionResult(
            action=ReactAction.TOOL_CALL,
            tool_name="crop_region",
            tool_arguments={"bbox": [0, 0, 10, 10]},
        )
        self.assertEqual(call_dec.action, ReactAction.TOOL_CALL)

        # TOOL_CALL without tool_name -> forbidden
        with self.assertRaises(ValueError):
            ReactDecisionResult(action=ReactAction.TOOL_CALL, tool_name=None)

        # FINISH with tool_name -> forbidden
        with self.assertRaises(ValueError):
            ReactDecisionResult(action=ReactAction.FINISH, tool_name="crop_region")

    def test_mock_model_caller_methods(self):
        caller = MockHJLModelCaller()
        ev_state = EvidenceState()
        ev_res = caller.verify_evidence(ev_state, "metal", 0.75, 0.20)
        self.assertEqual(ev_res.status, EvidenceStatus.FAIL)
        self.assertEqual(ev_res.conclusion, EvidenceConclusion.UNRESOLVED)


if __name__ == "__main__":
    unittest.main()
