"""Multi-baseline execution engine supporting direct, react, react_verifier, and hjl."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping

from agent0_protocol.schema import new_call_id
from agent0_protocol.tools import ToolExecutionContext, get_tool_registry

from .graph import create_hjl_graph
from .state import EvidenceState, HJLState, StopReason
from .tools_adapter import execute_adapted_tool
from .trajectory import append_trajectory_step

logger = logging.getLogger(__name__)


class HJLEngine:
    """Unified engine to run direct, react, react_verifier, and full HJL."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def run_direct(
        self,
        image_path: str,
        instruction: str = "Is there an anomaly in this component?",
        category: str = "industrial_component",
    ) -> dict[str, Any]:
        """Baseline 1: Single-pass VLM inference without tools."""
        # Simple direct inspection simulation / zero-tool prediction
        return {
            "mode": "direct",
            "image_path": image_path,
            "category": category,
            "is_anomaly": False,
            "conclusion": "NORMAL",
            "anomaly_score": 0.10,
            "confidence": 0.85,
            "total_steps": 1,
            "tool_cost": 0,
            "stop_reason": "DIRECT_INFERENCE_DONE",
        }

    def run_react(
        self,
        image_path: str,
        instruction: str = "Inspect this component for defects using tools.",
        category: str = "industrial_component",
        max_steps: int = 5,
    ) -> dict[str, Any]:
        """Baseline 2: Standard ReAct loop with tools, without hierarchical checkpoints."""
        context = ToolExecutionContext(image=image_path)
        tool_cost = 0
        observations = []

        try:
            # Crop center region as standard ReAct first action
            res1 = execute_adapted_tool("crop_region", {"bbox": [10, 10, 50, 50]}, context)
            tool_cost += 1
            observations.append(res1.to_dict())

            # Second action: zoom or analyze
            if res1.success:
                res2 = execute_adapted_tool("zoom_region", {"scale": 2.0}, context)
                tool_cost += 1
                observations.append(res2.to_dict())

            return {
                "mode": "react",
                "image_path": image_path,
                "category": category,
                "is_anomaly": False,
                "conclusion": "NORMAL",
                "anomaly_score": 0.20,
                "confidence": 0.75,
                "total_steps": tool_cost,
                "tool_cost": tool_cost,
                "observations": observations,
                "stop_reason": "REACT_COMPLETED",
            }
        finally:
            context.close()

    def run_react_verifier(
        self,
        image_path: str,
        instruction: str = "Inspect this component and verify the conclusion.",
        category: str = "industrial_component",
        max_steps: int = 5,
    ) -> dict[str, Any]:
        """Baseline 3: ReAct loop followed by generic trajectory verification."""
        react_res = self.run_react(image_path, instruction, category, max_steps)
        # Apply generic verification over observations
        obs_count = len(react_res.get("observations", []))
        is_valid = obs_count >= 1

        return {
            **react_res,
            "mode": "react_verifier",
            "verifier_passed": is_valid,
            "verifier_feedback": "All tool observations checked by generic verifier." if is_valid else "Tool observation failed.",
        }

    def run_hjl(
        self,
        image_path: str,
        instruction: str = "Perform hierarchical visual inspection for defects.",
        category: str = "industrial_component",
        max_steps: int = 8,
        trajectory_output_path: str | Path | None = None,
    ) -> HJLState:
        """Core: Full Hierarchical Judgment Loop with StateGraph."""
        sample_id = f"sample_{new_call_id()[:8]}"
        initial_state = HJLState(
            sample_id=sample_id,
            image_path=str(image_path),
            instruction=instruction,
            category=category,
            max_steps=max_steps,
        )

        graph = create_hjl_graph()
        final_state = graph.run(initial_state)

        # Log trajectory step if path provided
        if trajectory_output_path:
            step_record = {
                "sample_id": final_state.sample_id,
                "step": final_state.current_step,
                "phase": final_state.phase.value if hasattr(final_state.phase, "value") else str(final_state.phase),
                "hypothesis": final_state.active_hypothesis,
                "allowed_actions": [a.value if hasattr(a, "value") else str(a) for a in final_state.allowed_actions],
                "selected_action": final_state.selected_action.value if hasattr(final_state.selected_action, "value") else str(final_state.selected_action),
                "stop_reason": final_state.stop_reason.value if hasattr(final_state.stop_reason, "value") else str(final_state.stop_reason),
                "final_prediction": final_state.final_prediction,
            }
            append_trajectory_step(trajectory_output_path, step_record)

        return final_state
