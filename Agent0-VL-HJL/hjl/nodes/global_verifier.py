"""Checkpoint 1: Global Verifier node for HJL."""

from __future__ import annotations

from typing import Any

from ..routing import GlobalRecoveryPolicy
from ..state import HJLPhase, HJLState, StopReason
from ..taxonomy import CheckpointJudgment, GlobalStatus


def global_verifier_node(
    state: HJLState,
    global_normal_confidence_threshold: float = 0.95,
) -> dict[str, Any]:
    """Evaluate whether global inspection identified valid candidate ROIs or confirmed normal."""
    is_normal = getattr(state, "global_is_normal", False)
    conf = float(getattr(state, "global_confidence", 0.80))

    # 1. High-Confidence Global Confirmed Normal Guard
    if (
        is_normal
        and conf >= global_normal_confidence_threshold
        and not state.candidate_regions
        and not state.evidence_state.unresolved_regions
    ):
        judgment = CheckpointJudgment(
            status=GlobalStatus.CONFIRMED_NORMAL,
            judgment_confidence=conf,
            reason=f"Global inspection confirms pristine surface without suspicious regions (confidence {conf:.2f} >= {global_normal_confidence_threshold:.2f}).",
        )
        return {
            "global_judgment": judgment,
            "stop_reason": StopReason.CONFIRMED_NORMAL,
        }

    # 2. Checkpoint FAIL -> Candidate Discovery Phase
    if not state.candidate_regions:
        judgment = CheckpointJudgment(
            status=GlobalStatus.FAIL,
            judgment_confidence=conf,
            reason="Global view is ambiguous or lacks clear candidate region proposals.",
        )
        allowed_actions = GlobalRecoveryPolicy.get_recovery_actions(GlobalStatus.FAIL)
        return {
            "global_judgment": judgment,
            "phase": HJLPhase.GLOBAL_DISCOVERY,
            "allowed_actions": allowed_actions,
        }

    # 3. Checkpoint PASS -> Hypothesis Inspection Phase
    judgment = CheckpointJudgment(
        status=GlobalStatus.PASS,
        judgment_confidence=conf,
        reason=f"Identified {len(state.candidate_regions)} candidate region(s) ready for regional inspection.",
    )
    return {
        "global_judgment": judgment,
        "phase": HJLPhase.HYPOTHESIS_INSPECTION,
    }
