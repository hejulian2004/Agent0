"""Unit tests for typed model caller schemas, validation helpers, and invariants."""

from __future__ import annotations

import unittest

from hjl.model_caller import (
    CandidateRegion,
    DirectPredictionResult,
    EvidenceVerificationResult,
    GenericVerificationResult,
    MockHJLModelCaller,
    ReactAction,
    ReactDecisionResult,
    ScriptedHJLModelCaller,
    require_bool,
    require_dict,
    require_float,
    require_list,
    require_str,
)
from hjl.state import EvidenceState
from hjl.taxonomy import EvidenceConclusion, EvidenceStatus


class TestHJLModelCaller(unittest.TestCase):
    def test_validation_helpers_primitive_enforcement(self):
        # require_bool: True/False are valid
        self.assertTrue(require_bool({"a": True}, "a"))
        self.assertFalse(require_bool({"a": False}, "a"))

        # String "false" or integer 0 must NOT be coerced to boolean!
        with self.assertRaises(ValueError):
            require_bool({"a": "false"}, "a")
        with self.assertRaises(ValueError):
            require_bool({"a": 0}, "a")
        with self.assertRaises(ValueError):
            require_bool({}, "a")

        # require_float: numeric in [min, max]
        self.assertEqual(require_float({"f": 0.85}, "f", 0.0, 1.0), 0.85)
        self.assertEqual(require_float({"f": 1}, "f", 0.0, 1.0), 1.0)
        # Out of bounds
        with self.assertRaises(ValueError):
            require_float({"f": 1.5}, "f", 0.0, 1.0)
        # Boolean must not be coerced to float
        with self.assertRaises(ValueError):
            require_float({"f": True}, "f", 0.0, 1.0)

        # require_str
        self.assertEqual(require_str({"s": " hello "}, "s"), "hello")
        with self.assertRaises(ValueError):
            require_str({"s": "   "}, "s")
        with self.assertRaises(ValueError):
            require_str({"s": 123}, "s")

        # require_dict and require_list
        self.assertEqual(require_dict({"d": {"k": 1}}, "d"), {"k": 1})
        with self.assertRaises(ValueError):
            require_dict({"d": [1, 2]}, "d")

        self.assertEqual(require_list({"l": [1, 2]}, "l"), [1, 2])
        with self.assertRaises(ValueError):
            require_list({"l": "abc"}, "l")

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

    def test_direct_prediction_cross_field_invariants(self):
        # Valid anomaly prediction
        pred_anom = DirectPredictionResult(
            is_anomaly=True,
            conclusion="ANOMALY",
            anomaly_score=0.9,
            confidence=0.95,
            explanation="Crack detected",
        )
        self.assertTrue(pred_anom.is_anomaly)

        # Valid normal prediction
        pred_norm = DirectPredictionResult(
            is_anomaly=False,
            conclusion="NORMAL",
            anomaly_score=0.1,
            confidence=0.90,
            explanation="Pristine surface",
        )
        self.assertFalse(pred_norm.is_anomaly)

        # Conflict: conclusion ANOMALY with is_anomaly=False
        with self.assertRaises(ValueError):
            DirectPredictionResult(
                is_anomaly=False,
                conclusion="ANOMALY",
                anomaly_score=0.9,
                confidence=0.95,
                explanation="Contradictory",
            )

        # Conflict: conclusion NORMAL with is_anomaly=True
        with self.assertRaises(ValueError):
            DirectPredictionResult(
                is_anomaly=True,
                conclusion="NORMAL",
                anomaly_score=0.1,
                confidence=0.95,
                explanation="Contradictory",
            )

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
            ReactDecisionResult(action=ReactAction.TOOL_CALL, tool_name=None, tool_arguments={})

        # TOOL_CALL without tool_arguments -> forbidden
        with self.assertRaises(ValueError):
            ReactDecisionResult(action=ReactAction.TOOL_CALL, tool_name="crop_region", tool_arguments=None)

        # TOOL_CALL with is_anomaly specified -> forbidden
        with self.assertRaises(ValueError):
            ReactDecisionResult(
                action=ReactAction.TOOL_CALL,
                tool_name="crop_region",
                tool_arguments={},
                is_anomaly=False,
            )

        # Valid FINISH
        fin_dec = ReactDecisionResult(
            action=ReactAction.FINISH,
            final_answer="Normal part",
            is_anomaly=False,
        )
        self.assertEqual(fin_dec.action, ReactAction.FINISH)

        # FINISH with tool_name -> forbidden
        with self.assertRaises(ValueError):
            ReactDecisionResult(action=ReactAction.FINISH, tool_name="crop_region", is_anomaly=False)

        # FINISH with non-boolean is_anomaly -> forbidden
        with self.assertRaises(ValueError):
            ReactDecisionResult(action=ReactAction.FINISH, final_answer="Done", is_anomaly=None)

    def test_scripted_model_caller(self):
        scripted_dec = ReactDecisionResult(action=ReactAction.FINISH, is_anomaly=True, final_answer="Defect found")
        caller = ScriptedHJLModelCaller(react_decision=scripted_dec)
        res = caller.react_step([], "dummy.png", ["crop_region"])
        self.assertEqual(res.action, ReactAction.FINISH)
        self.assertTrue(res.is_anomaly)
        self.assertIn("react_step", caller.call_history)


if __name__ == "__main__":
    unittest.main()
