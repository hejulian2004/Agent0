"""Checkpoint 1: Global Verifier node for HJL."""

from __future__ import annotations

from typing import Any

from ..routing import GlobalRecoveryPolicy
from ..state import HJLPhase, HJLState, StopReason
from ..taxonomy import CheckpointJudgment, GlobalStatus


def global_verifier_node(state: HJLState) -> dict[str, Any]:
    """Evaluate whether global inspection identified valid candidate ROIs or confirmed normal."""
    # Check if forced normal by instruction/category or clean image
    if state.instruction and "force_normal" in state.instruction.lower():
        judgment = CheckpointJudgment(
            status=GlobalStatus.CONFIRMED_NORMAL,
            judgment_confidence=0.98,
            reason="Global inspection confirms pristine surface without suspicious regions.",
        )
        return {
            "global_judgment": judgment,
            "stop_reason": StopReason.CONFIRMED_NORMAL,
        }

    # If no candidate regions were proposed or observation is ambiguous
    if not state.candidate_regions:
        judgment = CheckpointJudgment(
            status=GlobalStatus.FAIL,
            judgment_confidence=0.60,
            reason="Global view is ambiguous or lacks clear candidate region proposals.",
        )
        allowed_actions = GlobalRecoveryPolicy.get_recovery_actions(GlobalStatus.FAIL)
        return {
            "global_judgment": judgment,
            "phase": HJLPhase.GLOBAL_DISCOVERY,
            "allowed_actions": allowed_actions,
        }

    # Candidate regions exist -> proceed to hypothesis inspection
    judgment = CheckpointJudgment(
        status=GlobalStatus.PASS,
        judgment_confidence=0.85,
        reason=f"Identified {len(state.candidate_regions)} candidate region(s) ready for regional inspection.",
    )
    return {
        "global_judgment": judgment,
        "phase": HJLPhase.HYPOTHESIS_INSPECTION,
    }
