"""Tool execution node for HJL."""

from __future__ import annotations

import copy
from typing import Any

from agent0_protocol.tools import ToolExecutionContext, get_tool_registry

from ..routing import FailureRoutingPolicy
from ..state import HJLState
from ..taxonomy import ActionType, FailureDiagnosis, FailureType
from ..tools_adapter import ToolResult, execute_adapted_tool


def tool_executor_node(
    state: HJLState,
    context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """Execute the planned tool call and handle execution success or failure."""
    tool_calls = state.tool_calls
    if not tool_calls:
        return {}

    call_spec = tool_calls[0]
    name = call_spec.get("name", "")
    arguments = call_spec.get("arguments", {})

    # Create or reuse execution context
    exec_context = context
    if exec_context is None:
        exec_context = ToolExecutionContext(image=state.image_path)

    # Save checkpoint before tool call for potential rollback
    exec_context.save_checkpoint()

    # Increment step counter and tool cost
    current_step = state.current_step + 1
    state.evidence_state.tool_cost += 1

    tool_result: ToolResult = execute_adapted_tool(
        name=name,
        arguments=arguments,
        context=exec_context,
        registry=get_tool_registry(),
    )

    observations = list(state.observations)

    # 1. Immediate Tool Execution Failure path (bypasses LLM diagnoser)
    if not tool_result.success:
        consecutive_failures = state.consecutive_tool_failures + 1
        error_msg = tool_result.error or f"Execution failed for tool {name}"

        # Roll back context to uncorrupted image state
        exec_context.rollback()

        # Deterministic TOOL_FAILURE assignment
        diagnosis = FailureDiagnosis(
            failure_type=FailureType.TOOL_FAILURE,
            cause=error_msg,
            diagnosis_confidence=1.0,
        )
        allowed_actions = FailureRoutingPolicy.get_allowed_actions(FailureType.TOOL_FAILURE)

        return {
            "current_step": current_step,
            "consecutive_tool_failures": consecutive_failures,
            "failure_type": FailureType.TOOL_FAILURE,
            "failure_reason": error_msg,
            "allowed_actions": allowed_actions,
            "selected_action": ActionType.RETRY_TOOL,
        }

    # 2. Tool Execution Success path -> Commit checkpoint (discard intermediate snapshot)
    if hasattr(exec_context, "_checkpoints") and exec_context._checkpoints:
        exec_context._checkpoints.pop()

    consecutive_failures = 0
    obs = {
        "step": current_step,
        "tool": name,
        "arguments": copy.deepcopy(arguments),
        "output_path": tool_result.output_path,
        "metadata": copy.deepcopy(tool_result.metadata),
    }
    observations.append(obs)

    return {
        "current_step": current_step,
        "consecutive_tool_failures": consecutive_failures,
        "observations": observations,
        "failure_type": None,
        "failure_reason": None,
    }
