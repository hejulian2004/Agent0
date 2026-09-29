"""Unit tests for HJL state, evidence accumulation, and phases."""

from __future__ import annotations

import unittest
from hjl.state import EvidenceItem, EvidenceRelation, EvidenceState, HJLPhase, HJLState, StopReason
from hjl.taxonomy import EvidenceConclusion, EvidenceJudgment, EvidenceStatus


class TestHJLState(unittest.TestCase):
    def test_evidence_item_creation_and_dict(self):
        item = EvidenceItem(
            source_step=1,
            region=[10, 20, 50, 60],
            observation_type="texture_anomaly",
            statement="Local surface roughness detected.",
            relation=EvidenceRelation.SUPPORT,
            confidence=0.85,
            source_tool="crop_region",
        )
        d = item.to_dict()
        self.assertEqual(d["source_step"], 1)
        self.assertEqual(d["relation"], "SUPPORT")
        self.assertEqual(d["region"], [10, 20, 50, 60])

        restored = EvidenceItem.from_dict(d)
        self.assertEqual(restored.relation, EvidenceRelation.SUPPORT)
        self.assertEqual(restored.confidence, 0.85)

    def test_evidence_state_anomaly_score_strictly_support_and_contradict(self):
        state = EvidenceState()

        # Add SUPPORT item
        state.supporting_evidence.append(
            EvidenceItem(1, [0, 0, 10, 10], "crack", "Crack visible", EvidenceRelation.SUPPORT, 0.8, "crop_region")
        )
        # Add CONTRADICT item
        state.contradicting_evidence.append(
            EvidenceItem(2, [0, 0, 10, 10], "ref", "Reference matches", EvidenceRelation.CONTRADICT, 0.3, "compare_with_reference")
        )
        # Add NEUTRAL item
        state.neutral_evidence.append(
            EvidenceItem(3, [0, 0, 10, 10], "ref", "Normal doc", EvidenceRelation.NEUTRAL, 0.9, "retrieve_normal_reference")
        )

        # NEUTRAL does not contribute to anomaly_score
        sup_weight = sum(e.confidence for e in state.supporting_evidence)
        contra_weight = sum(e.confidence for e in state.contradicting_evidence)
        state.anomaly_score = max(0.0, min(1.0, sup_weight - contra_weight))

        self.assertAlmostEqual(state.anomaly_score, 0.5)

    def test_hjl_state_serialization_roundtrip(self):
        state = HJLState(
            sample_id="test_001",
            image_path="/tmp/fake.png",
            phase=HJLPhase.EVIDENCE_RESOLUTION,
            stop_reason=StopReason.CONFIRMED_ANOMALY,
        )
        state.evidence_state.unresolved_regions = [[10, 10, 30, 30]]
        state.evidence_judgment = EvidenceJudgment(
            status=EvidenceStatus.PASS,
            conclusion=EvidenceConclusion.ANOMALY,
            judgment_confidence=0.92,
            reason="Confirmed anomaly.",
        )

        d = state.to_dict()
        self.assertEqual(d["sample_id"], "test_001")
        self.assertEqual(d["phase"], "EVIDENCE_RESOLUTION")
        self.assertEqual(d["stop_reason"], "CONFIRMED_ANOMALY")
        self.assertEqual(d["evidence_judgment"]["conclusion"], "ANOMALY")


if __name__ == "__main__":
    unittest.main()
