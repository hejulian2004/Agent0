"""Finalizer node for HJL.

Synthesizes the final prediction structure directly from verified Evidence or Global judgments
without performing secondary or independent anomaly re-reasoning.
"""

from __future__ import annotations

from typing import Any

from ..state import HJLState, StopReason
from ..taxonomy import EvidenceConclusion, EvidenceStatus, GlobalStatus


def finalizer_node(state: HJLState) -> dict[str, Any]:
    """Formulate the final prediction structure from verified judgments."""
    is_anomaly = False
    conclusion = "NORMAL"
    conf = 0.5
    stop_reason = state.stop_reason

    # 1. From definitive Evidence Judgment
    if state.evidence_judgment is not None and state.evidence_judgment.status == EvidenceStatus.PASS:
        conf = state.evidence_judgment.judgment_confidence
        if state.evidence_judgment.conclusion == EvidenceConclusion.ANOMALY:
            is_anomaly = True
            conclusion = "ANOMALY"
            stop_reason = StopReason.CONFIRMED_ANOMALY
        elif state.evidence_judgment.conclusion == EvidenceConclusion.NORMAL:
            is_anomaly = False
            conclusion = "NORMAL"
            stop_reason = StopReason.CONFIRMED_NORMAL

    # 2. From Global Confirmed Normal
    elif state.global_judgment is not None and state.global_judgment.status == GlobalStatus.CONFIRMED_NORMAL:
        is_anomaly = False
        conclusion = "NORMAL"
        conf = state.global_judgment.judgment_confidence
        stop_reason = StopReason.CONFIRMED_NORMAL

    # 3. From budget / fallback termination
    else:
        # Best effort based on accumulated anomaly score
        is_anomaly = state.evidence_state.anomaly_score >= 0.5
        conclusion = "ANOMALY" if is_anomaly else "NORMAL"
        conf = max(state.evidence_state.anomaly_score, 1.0 - state.evidence_state.anomaly_score)
        if stop_reason is None:
            stop_reason = StopReason.MAX_STEPS

    # Collect detected defect regions from supporting evidence
    detected_regions = [
        e.region for e in state.evidence_state.supporting_evidence
        if e.region is not None
    ]

    final_prediction = {
        "sample_id": state.sample_id,
        "is_anomaly": is_anomaly,
        "conclusion": conclusion,
        "anomaly_score": round(state.evidence_state.anomaly_score, 4),
        "confidence": round(conf, 4),
        "detected_regions": detected_regions,
        "inspected_regions": state.evidence_state.inspected_regions,
        "evidence_summary": [e.statement for e in state.evidence_state.evidence_items],
        "stop_reason": stop_reason.value if stop_reason else StopReason.CONFIRMED_ANOMALY.value,
        "total_steps": state.current_step,
        "tool_cost": state.evidence_state.tool_cost,
    }

    return {
        "stop_reason": stop_reason,
        "final_prediction": final_prediction,
    }
