"""Unit tests for HJL taxonomy, checkpoint judgments, and failure routing policies."""

from __future__ import annotations

import unittest
from hjl.policies.failure_policy import FailurePolicy
from hjl.routing import FailureRoutingPolicy, GlobalRecoveryPolicy
from hjl.taxonomy import (
    ActionType,
    CheckpointJudgment,
    EvidenceConclusion,
    EvidenceJudgment,
    EvidenceStatus,
    FailureDiagnosis,
    FailureType,
    GlobalStatus,
    RegionalStatus,
    SEMANTIC_FAILURE_TYPES,
)


class TestHJLTaxonomyRouting(unittest.TestCase):
    def test_checkpoint_judgment_schemas(self):
        cj = CheckpointJudgment(status=GlobalStatus.PASS, judgment_confidence=0.88, reason="Valid candidate")
        d = cj.to_dict()
        self.assertEqual(d["status"], "PASS")
        self.assertEqual(d["judgment_confidence"], 0.88)

        ej = EvidenceJudgment(
            status=EvidenceStatus.PASS,
            conclusion=EvidenceConclusion.ANOMALY,
            judgment_confidence=0.95,
            reason="High anomaly score",
        )
        ed = ej.to_dict()
        self.assertEqual(ed["status"], "PASS")
        self.assertEqual(ed["conclusion"], "ANOMALY")

    def test_semantic_failure_types_count(self):
        # 8 total failure types: 1 deterministic TOOL_FAILURE + 7 semantic failure types
        self.assertEqual(len(FailureType), 8)
        self.assertEqual(len(SEMANTIC_FAILURE_TYPES), 7)
        self.assertNotIn(FailureType.TOOL_FAILURE, SEMANTIC_FAILURE_TYPES)

    def test_failure_routing_policy_action_masks(self):
        # Each failure type must produce a valid non-empty Action Mask
        for ft in FailureType:
            actions = FailureRoutingPolicy.get_allowed_actions(ft)
            self.assertTrue(len(actions) > 0, f"FailureType {ft} has empty action mask")
            for act in actions:
                self.assertIsInstance(act, ActionType)

        # Verify specific mappings
        self.assertEqual(FailureRoutingPolicy.get_allowed_actions(FailureType.TOOL_FAILURE), [ActionType.RETRY_TOOL])
        self.assertEqual(FailureRoutingPolicy.get_allowed_actions(FailureType.LOW_RESOLUTION), [ActionType.ENHANCE_REGION])
        self.assertEqual(FailureRoutingPolicy.get_allowed_actions(FailureType.MISSING_REFERENCE), [ActionType.RETRIEVE_REFERENCE])

    def test_global_recovery_policy(self):
        actions = GlobalRecoveryPolicy.get_recovery_actions(GlobalStatus.FAIL)
        self.assertIn(ActionType.GLOBAL_SCAN, actions)
        self.assertIn(ActionType.RELOCALIZE, actions)

        # PASS should not require recovery
        self.assertEqual(GlobalRecoveryPolicy.get_recovery_actions(GlobalStatus.PASS), [])

    def test_failure_policy_escalation(self):
        policy = FailurePolicy(max_repeated_failures=2)
        policy.record_failure(FailureType.LOW_RESOLUTION)
        self.assertFalse(policy.is_escalated(FailureType.LOW_RESOLUTION))
        policy.record_failure(FailureType.LOW_RESOLUTION)
        self.assertFalse(policy.is_escalated(FailureType.LOW_RESOLUTION))
        policy.record_failure(FailureType.LOW_RESOLUTION)
        # 3rd time exceeds max_repeated_failures=2
        self.assertTrue(policy.is_escalated(FailureType.LOW_RESOLUTION))


if __name__ == "__main__":
    unittest.main()
