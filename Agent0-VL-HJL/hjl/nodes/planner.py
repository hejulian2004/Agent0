"""Action planning node for HJL."""

from __future__ import annotations

import copy
from typing import Any

from ..state import HJLPhase, HJLState, StopReason
from ..taxonomy import ActionType


def planner_node(
    state: HJLState,
    reference_corpus_dir: str | None = None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Plan concrete tool call based on current phase and allowed action mask."""
    phase = state.phase
    allowed_actions = list(state.allowed_actions)
    tool_calls: list[dict[str, Any]] = []
    selected_action: ActionType | None = state.selected_action

    # 1. Global Discovery Mode (hypothesis-free candidate discovery)
    if phase == HJLPhase.GLOBAL_DISCOVERY or state.active_hypothesis is None:
        if not allowed_actions:
            allowed_actions = [ActionType.GLOBAL_SCAN, ActionType.RELOCALIZE]

        selected_action = allowed_actions[0]
        tool_calls.append({
            "name": "localize_candidate",
            "arguments": {"use_original": True},
        })

        plan = {
            "phase": phase.value,
            "action": selected_action.value,
            "intent": "Discover and localize candidate suspicious regions on global image.",
            "tool_call": tool_calls[0],
        }

    # 2. Evidence Resolution Mode (resolving contradictions or missing normal reference)
    elif phase == HJLPhase.EVIDENCE_RESOLUTION:
        if not allowed_actions:
            allowed_actions = [ActionType.RETRIEVE_REFERENCE, ActionType.CROSS_VALIDATE]

        selected_action = state.selected_action or allowed_actions[0]

        if selected_action == ActionType.RETRIEVE_REFERENCE:
            category = state.category or "industrial_component"
            ret_args: dict[str, Any] = {
                "category": category,
                "allow_synthetic": allow_synthetic,
            }
            if reference_corpus_dir:
                ret_args["corpus_dir"] = reference_corpus_dir
            tool_calls.append({
                "name": "retrieve_normal_reference",
                "arguments": ret_args,
            })
        elif selected_action == ActionType.CROSS_VALIDATE:
            ref_path = (
                state.evidence_state.normal_references[-1]["reference_path"]
                if state.evidence_state.normal_references
                else None
            )
            comp_args: dict[str, Any] = {
                "reference_path": ref_path,
                "normalized_bbox": state.active_region_normalized_bbox,
                "rotation_deg": state.active_region_rotation_deg,
            }
            if state.active_region_original_bbox:
                comp_args["bbox"] = list(state.active_region_original_bbox)
            tool_calls.append({
                "name": "compare_with_reference",
                "arguments": comp_args if ref_path else {},
            })
        elif selected_action == ActionType.ENHANCE_REGION:
            zoom_args: dict[str, Any] = {"scale": 2.0}
            if state.active_region_original_bbox:
                zoom_args["bbox"] = list(state.active_region_original_bbox)
            tool_calls.append({
                "name": "zoom_region",
                "arguments": zoom_args,
            })
        elif selected_action == ActionType.INSPECT_NEXT_REGION:
            next_bbox = (
                state.evidence_state.unresolved_regions[0]
                if state.evidence_state.unresolved_regions
                else [0, 0, 10, 10]
            )
            tool_calls.append({
                "name": "crop_region",
                "arguments": {"bbox": next_bbox, "use_original": True},
            })
        elif selected_action == ActionType.RETRY_TOOL:
            if state.last_failed_tool_call is not None:
                tool_calls.append(copy.deepcopy(state.last_failed_tool_call))
            else:
                return {"stop_reason": StopReason.NO_VALID_ACTION}
        else:
            tool_calls.append({
                "name": "crop_region",
                "arguments": {"bbox": [0, 0, 20, 20], "use_original": True},
            })

        plan = {
            "phase": phase.value,
            "action": selected_action.value,
            "intent": f"Resolve pending evidence questions via {selected_action.value}.",
            "tool_call": tool_calls[0],
        }

    # 3. Hypothesis Inspection Mode (standard focused inspection)
    else:
        if not allowed_actions:
            allowed_actions = [ActionType.ENHANCE_REGION, ActionType.RELOCALIZE]

        selected_action = state.selected_action or allowed_actions[0]

        # Determine target region (in original-image space)
        target_bbox = None
        if state.evidence_state.unresolved_regions:
            target_bbox = state.evidence_state.unresolved_regions[0]
        elif state.active_hypothesis and state.active_hypothesis.get("target_region"):
            target_bbox = state.active_hypothesis["target_region"]
        elif state.candidate_regions and "bbox" in state.candidate_regions[0]:
            target_bbox = state.candidate_regions[0]["bbox"]
        else:
            target_bbox = [0, 0, 50, 50]

        if selected_action == ActionType.ENHANCE_REGION:
            # If region was already cropped, zoom; else crop from original image
            if state.observations and any(o.get("tool") == "crop_region" for o in state.observations):
                zoom_args: dict[str, Any] = {"scale": 2.0}
                if state.active_region_original_bbox:
                    zoom_args["bbox"] = list(state.active_region_original_bbox)
                tool_calls.append({
                    "name": "zoom_region",
                    "arguments": zoom_args,
                })
            else:
                tool_calls.append({
                    "name": "crop_region",
                    "arguments": {"bbox": target_bbox, "use_original": True},
                })
        elif selected_action == ActionType.CROSS_VALIDATE:
            ref_path = (
                state.evidence_state.normal_references[-1]["reference_path"]
                if state.evidence_state.normal_references
                else None
            )
            comp_args: dict[str, Any] = {
                "reference_path": ref_path,
                "normalized_bbox": state.active_region_normalized_bbox,
                "rotation_deg": state.active_region_rotation_deg,
            }
            if state.active_region_original_bbox:
                comp_args["bbox"] = list(state.active_region_original_bbox)
            tool_calls.append({
                "name": "compare_with_reference",
                "arguments": comp_args if ref_path else {},
            })
        elif selected_action == ActionType.RETRIEVE_REFERENCE:
            ret_args2: dict[str, Any] = {
                "category": state.category or "industrial_component",
                "allow_synthetic": allow_synthetic,
            }
            if reference_corpus_dir:
                ret_args2["corpus_dir"] = reference_corpus_dir
            tool_calls.append({
                "name": "retrieve_normal_reference",
                "arguments": ret_args2,
            })
        elif selected_action == ActionType.RELOCALIZE:
            tool_calls.append({
                "name": "localize_candidate",
                "arguments": {"use_original": True},
            })
        elif selected_action == ActionType.ALIGN_VIEW:
            tool_calls.append({
                "name": "rotate_image",
                "arguments": {"angle": 90.0},
            })
        elif selected_action == ActionType.RETRY_TOOL:
            if state.last_failed_tool_call is not None:
                tool_calls.append(copy.deepcopy(state.last_failed_tool_call))
            else:
                return {"stop_reason": StopReason.NO_VALID_ACTION}
        else:
            tool_calls.append({
                "name": "crop_region",
                "arguments": {"bbox": target_bbox, "use_original": True},
            })

        plan = {
            "phase": phase.value,
            "action": selected_action.value,
            "intent": f"Inspect target region {target_bbox} for {state.active_hypothesis.get('type') if state.active_hypothesis else 'defect'}.",
            "tool_call": tool_calls[0],
        }

    plan_history = list(state.plan_history)
    plan_history.append(plan)

    return {
        "current_plan": plan,
        "plan_history": plan_history,
        "tool_calls": tool_calls,
        "allowed_actions": allowed_actions,
        "selected_action": selected_action,
    }
