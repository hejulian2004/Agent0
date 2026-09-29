"""Stopping policy for HJL evaluating evidence-aware and budget termination."""

from __future__ import annotations

from typing import Any, Mapping

from ..state import HJLState, StopReason
from ..taxonomy import EvidenceConclusion, EvidenceStatus, GlobalStatus


class StoppingPolicy:
    """Evaluates whether HJL should terminate based on budget or conclusive evidence."""

    @staticmethod
    def evaluate(state: HJLState, config: Mapping[str, Any] | None = None) -> StopReason | None:
        cfg = config or {}
        max_steps = int(cfg.get("max_steps", state.max_steps))
        tool_failure_limit = int(cfg.get("tool_failure_limit", 3))

        # 1. Budget constraint
        if state.current_step >= max_steps:
            return StopReason.MAX_STEPS

        # 2. Tool failure limit
        if state.consecutive_tool_failures >= tool_failure_limit:
            return StopReason.TOOL_FAILURE_LIMIT

        # 3. Global early-exit normal
        if (
            state.global_judgment is not None
            and state.global_judgment.status == GlobalStatus.CONFIRMED_NORMAL
        ):
            return StopReason.CONFIRMED_NORMAL

        # 4. Evidence definitive verdict
        if state.evidence_judgment is not None and state.evidence_judgment.status == EvidenceStatus.PASS:
            if state.evidence_judgment.conclusion == EvidenceConclusion.ANOMALY:
                return StopReason.CONFIRMED_ANOMALY
            elif state.evidence_judgment.conclusion == EvidenceConclusion.NORMAL:
                return StopReason.CONFIRMED_NORMAL

        return None
