"""Checkpoint 3: Evidence Verifier node for HJL."""

from __future__ import annotations

from typing import Any

from ..state import EvidenceRelation, HJLPhase, HJLState
from ..taxonomy import EvidenceConclusion, EvidenceJudgment, EvidenceStatus


def evidence_verifier_node(
    state: HJLState,
    anomaly_threshold: float = 0.75,
    normal_threshold: float = 0.20,
    checkpoint_confidence_threshold: float = 0.80,
    min_evidence_count: int = 1,
) -> dict[str, Any]:
    """Evaluate whether accumulated persistent evidence is sufficient for a definitive decision."""
    ev = state.evidence_state
    score = ev.anomaly_score

    # 1. Definitive Anomaly Check
    if score >= anomaly_threshold:
        judgment = EvidenceJudgment(
            status=EvidenceStatus.PASS,
            conclusion=EvidenceConclusion.ANOMALY,
            judgment_confidence=checkpoint_confidence_threshold,
            reason=f"Accumulated evidence definitively confirms anomaly (anomaly_score={score:.2f} >= {anomaly_threshold}).",
        )
        return {"evidence_judgment": judgment}

    # 2. Definitive Normal Check with Strict Positive Evidence Guard
    has_explicit_normal = any(
        e.observation_type in {
            "normal_reference_match",
            "reference_comparison",
            "verified_normal_feature",
        }
        and e.relation == EvidenceRelation.CONTRADICT
        for e in ev.evidence_items
    )
    unresolved_clear = (len(ev.unresolved_regions) == 0 and len(ev.unresolved_questions) == 0)
    has_sufficient_records = len(ev.evidence_items) >= min_evidence_count

    if (
        score <= normal_threshold
        and unresolved_clear
        and has_sufficient_records
        and has_explicit_normal
    ):
        judgment = EvidenceJudgment(
            status=EvidenceStatus.PASS,
            conclusion=EvidenceConclusion.NORMAL,
            judgment_confidence=checkpoint_confidence_threshold,
            reason=f"Persistent evidence confirms normal component across all inspected regions (score={score:.2f}).",
        )
        return {"evidence_judgment": judgment}

    # 3. Unresolved / Gray Zone Check
    reason_parts = []
    if normal_threshold < score < anomaly_threshold:
        reason_parts.append(f"anomaly_score {score:.2f} lies in gray zone [{normal_threshold}, {anomaly_threshold}]")
    if ev.unresolved_regions:
        reason_parts.append(f"{len(ev.unresolved_regions)} candidate region(s) remain uninspected")
    if ev.unresolved_questions:
        reason_parts.append(f"{len(ev.unresolved_questions)} unresolved question(s)")
    if not has_explicit_normal:
        reason_parts.append("lacks explicit reference-consistent normal evidence")

    reason_text = "Evidence insufficient: " + "; ".join(reason_parts) if reason_parts else "Evidence is inconclusive."

    judgment = EvidenceJudgment(
        status=EvidenceStatus.FAIL,
        conclusion=EvidenceConclusion.UNRESOLVED,
        judgment_confidence=0.65,
        reason=reason_text,
    )

    return {
        "evidence_judgment": judgment,
        "phase": HJLPhase.EVIDENCE_RESOLUTION,
    }
