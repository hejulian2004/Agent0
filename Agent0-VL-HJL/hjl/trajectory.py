"""Trajectory logging, JSONL serialization, and CanonicalTrajectory export for HJL."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent0_protocol.schema import CanonicalTrajectory, new_call_id
from agent0_protocol.tools import get_tool_registry

from .state import HJLState


def append_trajectory_step(
    output_path: str | Path,
    record: dict[str, Any],
) -> None:
    """Append one step record to a JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_canonical_trajectory(state: HJLState) -> CanonicalTrajectory:
    """Convert an HJLState and its observation history to a valid CanonicalTrajectory."""
    registry = get_tool_registry()
    tools = registry.definitions()

    trajectory = CanonicalTrajectory(
        trajectory_id=f"traj_hjl_{state.sample_id}",
        tools=tools,
        metadata={
            "sample_id": state.sample_id,
            "category": state.category,
            "phase": state.phase.value if hasattr(state.phase, "value") else str(state.phase),
            "stop_reason": state.stop_reason.value if hasattr(state.stop_reason, "value") else str(state.stop_reason),
            "anomaly_score": state.evidence_state.anomaly_score,
            "total_steps": state.current_step,
            "tool_cost": state.evidence_state.tool_cost,
        },
    )

    # Initial user message
    trajectory.append({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": state.instruction},
        ],
    })

    # Convert observation history into function calls and outputs
    for i, obs in enumerate(state.observations):
        call_id = f"call_{state.sample_id}_{i+1}"
        tool_name = obs.get("tool", "crop_region")

        # Map to canonical tool name if adapted
        canonical_name = tool_name
        raw_args = copy.deepcopy(obs.get("arguments", {}))
        call_args = raw_args

        if tool_name == "crop_region":
            canonical_name = "crop_image"
            call_args = {"bbox": list(raw_args.get("bbox", [0, 0, 10, 10]))}
        elif tool_name == "zoom_region":
            canonical_name = "zoom_image"
            call_args = {"scale": float(raw_args.get("scale", 2.0))}
        elif tool_name == "retrieve_normal_reference":
            canonical_name = "retrieve"
            call_args = {"query": str(raw_args.get("query") or raw_args.get("category", "normal reference"))}
        elif tool_name == "rotate_image":
            canonical_name = "rotate_image"
            call_args = {"angle": float(raw_args.get("angle", 90.0))}
        else:
            if not any(t["name"] == canonical_name for t in tools):
                canonical_name = "visual_analyzer"
                call_args = {}
        trajectory.append({
            "type": "function_call",
            "call_id": call_id,
            "name": canonical_name,
            "arguments": copy.deepcopy(call_args),
        })

        output_data = {
            "success": True,
            "metadata": obs.get("metadata", {}),
        }
        trajectory.append({
            "type": "function_call_output",
            "call_id": call_id,
            "output": output_data,
        })

    # Final assistant message summarizing prediction
    conclusion = "NORMAL"
    if state.final_prediction:
        conclusion = state.final_prediction.get("conclusion", "NORMAL")

    final_text = (
        f"Conclusion: {conclusion}. Anomaly score: {state.evidence_state.anomaly_score:.2f}. "
        f"Stop reason: {state.stop_reason.value if state.stop_reason else 'DONE'}."
    )
    trajectory.append({
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": final_text},
        ],
    })

    trajectory.validate()
    return trajectory
