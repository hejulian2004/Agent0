"""Constrained Replanner node for HJL."""

from __future__ import annotations

from typing import Any

from ..policies.stopping_policy import StoppingPolicy
from ..state import HJLPhase, HJLState, StopReason
from ..taxonomy import ActionType


def replanner_node(state: HJLState) -> dict[str, Any]:
    """Select the optimal recovery action constrained by the allowed action mask."""
    # 1. Check stopping policy (e.g. max_steps, tool_failure_limit)
    stop_reason = StoppingPolicy.evaluate(state)
    if stop_reason is not None:
        return {"stop_reason": stop_reason}

    # 2. Check if action mask is empty
    if not state.allowed_actions:
        return {"stop_reason": StopReason.NO_VALID_ACTION}

    # 3. Select action from the allowed action set
    allowed = state.allowed_actions
    selected_action: ActionType = allowed[0]

    # Prioritize enhancement/relocalization based on failure context
    ft = state.failure_type
    ft_val = ft.value if hasattr(ft, "value") else str(ft)

    if ActionType.ENHANCE_REGION in allowed and ft_val == "LOW_RESOLUTION":
        selected_action = ActionType.ENHANCE_REGION
    elif ActionType.RELOCALIZE in allowed and ft_val == "WRONG_REGION":
        selected_action = ActionType.RELOCALIZE
    elif ActionType.RETRIEVE_REFERENCE in allowed and ft_val == "MISSING_REFERENCE":
        selected_action = ActionType.RETRIEVE_REFERENCE
    elif ActionType.CROSS_VALIDATE in allowed and ft_val == "CONTRADICTORY_EVIDENCE":
        selected_action = ActionType.CROSS_VALIDATE
    elif ActionType.RETRY_TOOL in allowed and ft_val == "TOOL_FAILURE":
        selected_action = ActionType.RETRY_TOOL
    else:
        selected_action = allowed[0]

    return {
        "selected_action": selected_action,
    }
