"""Tool execution node for HJL."""

from __future__ import annotations

import copy
from typing import Any

from PIL import Image

from agent0_protocol.tools import ToolExecutionContext, get_tool_registry

from ..state import HJLState, StopReason
from ..taxonomy import ActionType, FailureType
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

    if "original_image_path" not in exec_context:
        exec_context["original_image_path"] = str(state.image_path)

    # Save checkpoint before tool call for potential rollback
    checkpoint = exec_context.checkpoint()

    # Increment step counter and tool cost
    current_step = state.current_step + 1
    state.evidence_state.tool_cost += 1

    tool_result: ToolResult = execute_adapted_tool(
        name=name,
        arguments=arguments,
        context=exec_context,
        registry=get_tool_registry(),
    )

    # Always append to observation history (never overwrite)
    obs_record = {
        "step": current_step,
        "tool": name,
        "arguments": copy.deepcopy(arguments),
        "success": tool_result.success,
        "output_path": tool_result.output_path,
        "metadata": copy.deepcopy(tool_result.metadata),
        "error": tool_result.error,
        "retriable": tool_result.retriable,
    }
    new_observations = list(state.observations)
    new_observations.append(obs_record)

    # 1. Tool Execution Failure path
    if not tool_result.success:
        consecutive_failures = state.consecutive_tool_failures + 1
        error_msg = tool_result.error or f"Execution failed for tool {name}"

        # Roll back context to uncorrupted image state
        exec_context.rollback(checkpoint)

        if tool_result.retriable:
            return {
                "current_step": current_step,
                "consecutive_tool_failures": consecutive_failures,
                "observations": new_observations,
                "failure_type": FailureType.TOOL_FAILURE,
                "failure_reason": error_msg,
                "last_failed_tool_call": copy.deepcopy(call_spec),
                "allowed_actions": [ActionType.RETRY_TOOL],
                "selected_action": ActionType.RETRY_TOOL,
            }
        else:
            # Deterministic, non-retriable failure terminates immediately
            return {
                "current_step": current_step,
                "consecutive_tool_failures": consecutive_failures,
                "observations": new_observations,
                "failure_type": FailureType.TOOL_FAILURE,
                "failure_reason": error_msg,
                "last_failed_tool_call": None,
                "allowed_actions": [],
                "selected_action": None,
                "stop_reason": StopReason.NO_VALID_ACTION,
            }

    # 2. Tool Execution Success path -> Commit transform state purely via dict
    updates: dict[str, Any] = {
        "current_step": current_step,
        "consecutive_tool_failures": 0,
        "observations": new_observations,
        "failure_type": None,
        "failure_reason": None,
        "last_failed_tool_call": None,
    }

    if name == "crop_region":
        bbox = arguments.get("bbox", [])
        try:
            with Image.open(state.image_path) as orig_img:
                orig_w, orig_h = orig_img.size
        except Exception:
            orig_w, orig_h = 100, 100

        if len(bbox) == 4:
            updates["active_region_original_bbox"] = list(bbox)
            updates["active_region_normalized_bbox"] = [
                bbox[0] / orig_w,
                bbox[1] / orig_h,
                bbox[2] / orig_w,
                bbox[3] / orig_h,
            ]
        updates["active_region_rotation_deg"] = 0.0

    elif name == "rotate_image":
        angle = float(arguments.get("angle", 0.0))
        curr_rot = getattr(state, "active_region_rotation_deg", 0.0)
        updates["active_region_rotation_deg"] = (curr_rot + angle) % 360.0

    return updates
