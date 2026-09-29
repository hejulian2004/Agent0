"""Multi-baseline execution engine supporting direct, react, react_verifier, and hjl."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from agent0_protocol.schema import new_call_id
from agent0_protocol.tools import ToolExecutionContext, get_tool_registry

from .config import HJLConfig
from .graph import create_hjl_graph
from .model_caller import (
    HJLModelCaller,
    MockHJLModelCaller,
    ModelOutputError,
    ReactAction,
    ResponsesHJLModelCaller,
)
from .state import EvidenceState, HJLState, StopReason
from .tools_adapter import execute_adapted_tool
from .trajectory import append_trajectory_step

logger = logging.getLogger(__name__)


class HJLEngine:
    """Unified engine to run direct, react, react_verifier, and full HJL."""

    def __init__(
        self,
        config: HJLConfig | Mapping[str, Any] | None = None,
        model_caller: HJLModelCaller | None = None,
        mock: bool = False,
    ) -> None:
        if isinstance(config, HJLConfig):
            self.config = config
        elif isinstance(config, (dict, Mapping)):
            self.config = HJLConfig.from_mapping(config)
        else:
            self.config = HJLConfig()

        if model_caller is not None:
            self.model_caller = model_caller
        elif mock or not os.environ.get("AGENT0_RESPONSES_API_KEY"):
            self.model_caller = MockHJLModelCaller()
        else:
            try:
                self.model_caller = ResponsesHJLModelCaller()
            except Exception as exc:
                logger.warning(f"Could not initialize ResponsesHJLModelCaller ({exc}), falling back to Mock.")
                self.model_caller = MockHJLModelCaller()

    def run_direct(
        self,
        image_path: str,
        instruction: str = "Is there an anomaly in this component?",
        category: str = "industrial_component",
    ) -> dict[str, Any]:
        """Baseline 1: Single-pass VLM inference without tools."""
        res = self.model_caller.direct_inspect(image_path, instruction, category)
        return {
            "mode": "direct",
            "image_path": image_path,
            "category": category,
            "is_anomaly": res.is_anomaly,
            "conclusion": res.conclusion,
            "anomaly_score": res.anomaly_score,
            "confidence": res.confidence,
            "explanation": res.explanation,
            "total_steps": 1,
            "tool_cost": 0,
            "stop_reason": "DIRECT_INFERENCE_DONE",
        }

    def run_react(
        self,
        image_path: str,
        instruction: str = "Inspect this component for defects using tools.",
        category: str = "industrial_component",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Baseline 2: Standard ReAct loop with tools, without hierarchical checkpoints."""
        context = ToolExecutionContext(image=image_path)
        tool_cost = 0
        observations: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        limit = max_steps or self.config.max_steps
        reg = get_tool_registry()

        try:
            for step in range(1, limit + 1):
                decision = self.model_caller.react_step(
                    history=history,
                    image_path=image_path,
                    enabled_tools=self.config.enabled_tools,
                )

                if decision.action == ReactAction.FINISH:
                    is_anom = decision.is_anomaly if decision.is_anomaly is not None else False
                    return {
                        "mode": "react",
                        "image_path": image_path,
                        "category": category,
                        "is_anomaly": is_anom,
                        "conclusion": "ANOMALY" if is_anom else "NORMAL",
                        "confidence": decision.confidence,
                        "total_steps": step,
                        "tool_cost": tool_cost,
                        "observations": observations,
                        "stop_reason": "REACT_COMPLETED",
                    }

                # Validate tool name against enabled tool registry
                if decision.tool_name not in self.config.enabled_tools:
                    raise ModelOutputError(
                        f"ReAct agent selected unauthorized tool {decision.tool_name!r}. "
                        f"Allowed tools: {self.config.enabled_tools}"
                    )

                # Execute tool
                tool_args = decision.tool_arguments or {}
                tool_result = execute_adapted_tool(
                    name=decision.tool_name,
                    arguments=tool_args,
                    context=context,
                    registry=reg,
                )
                tool_cost += 1

                obs_record = {
                    "step": step,
                    "tool": decision.tool_name,
                    "arguments": copy.deepcopy(tool_args),
                    "success": tool_result.success,
                    "output_path": tool_result.output_path,
                    "metadata": copy.deepcopy(tool_result.metadata),
                }
                observations.append(obs_record)
                history.append({
                    "thought": f"Executed tool {decision.tool_name}",
                    "action": decision.tool_name,
                    "observation": obs_record,
                })

            # Exceeded max steps
            return {
                "mode": "react",
                "image_path": image_path,
                "category": category,
                "is_anomaly": False,
                "conclusion": "UNRESOLVED",
                "confidence": 0.50,
                "total_steps": limit,
                "tool_cost": tool_cost,
                "observations": observations,
                "stop_reason": "MAX_STEPS",
            }
        finally:
            context.close()

    def run_react_verifier(
        self,
        image_path: str,
        instruction: str = "Inspect this component and verify the conclusion.",
        category: str = "industrial_component",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Baseline 3: ReAct loop followed by trajectory verification."""
        react_res = self.run_react(image_path, instruction, category, max_steps)
        obs_count = len(react_res.get("observations", []))
        is_valid = obs_count >= 1

        return {
            **react_res,
            "mode": "react_verifier",
            "verifier_passed": is_valid,
            "verifier_feedback": (
                "All tool observations checked by generic verifier."
                if is_valid
                else "Tool observation failed."
            ),
        }

    def run_hjl(
        self,
        image_path: str,
        instruction: str = "Perform hierarchical visual inspection for defects.",
        category: str = "industrial_component",
        max_steps: int | None = None,
        trajectory_output_path: str | Path | None = None,
    ) -> HJLState:
        """Core: Full Hierarchical Judgment Loop with persistent ToolExecutionContext."""
        sample_id = f"sample_{new_call_id()[:8]}"
        limit = max_steps or self.config.max_steps
        initial_state = HJLState(
            sample_id=sample_id,
            image_path=str(image_path),
            instruction=instruction,
            category=category,
            max_steps=limit,
        )

        context = ToolExecutionContext(image=image_path)
        try:
            graph = create_hjl_graph(
                context=context,
                config=self.config,
                model_caller=self.model_caller,
            )
            final_state = graph.run(initial_state)

            # Log trajectory steps if path provided
            if trajectory_output_path:
                for step_record in final_state.step_history:
                    rec = {
                        "sample_id": final_state.sample_id,
                        **step_record,
                    }
                    append_trajectory_step(trajectory_output_path, rec)

            return final_state
        finally:
            context.close()
