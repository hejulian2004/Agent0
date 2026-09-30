"""Trajectory logging, JSONL serialization, and CanonicalTrajectory export for HJL."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent0_protocol.schema import CanonicalTrajectory, new_call_id

from .state import HJLState
from .tools_adapter import get_hjl_tool_definitions


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
    tools = get_hjl_tool_definitions()

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

    # Initial user message with visual context
    user_content: list[dict[str, Any]] = [
        {"type": "input_text", "text": state.instruction},
    ]
    if state.image_path:
        user_content.append({
            "type": "input_image",
            "image_url": str(state.image_path),
        })

    trajectory.append({
        "type": "message",
        "role": "user",
        "content": user_content,
    })

    # Faithfully convert observation history into function calls and outputs without label rewriting
    for i, obs in enumerate(state.observations):
        call_id = f"call_{state.sample_id}_{i+1}"
        tool_name = str(obs.get("tool", "crop_region"))
        raw_args = copy.deepcopy(obs.get("arguments", {}))

        trajectory.append({
            "type": "function_call",
            "call_id": call_id,
            "name": tool_name,
            "arguments": raw_args,
        })

        output_data = {
            "success": bool(obs.get("success", True)),
            "output_path": obs.get("output_path"),
            "metadata": copy.deepcopy(obs.get("metadata", {})),
            "error": obs.get("error"),
            "retriable": bool(obs.get("retriable", False)),
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
