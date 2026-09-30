"""Multi-baseline execution engine supporting direct, react, react_verifier, and hjl."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from agent0_protocol.schema import new_call_id
from agent0_protocol.tools import ToolExecutionContext, _current_image_path, get_tool_registry

from .config import HJLConfig
from .graph import create_hjl_graph
from .model_caller import (
    HJLModelCaller,
    MockHJLModelCaller,
    ReactAction,
    ReactDecisionResult,
    ResponsesHJLModelCaller,
)
from .state import HJLState
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

        self.mock = mock

        if model_caller is not None:
            self.model_caller = model_caller
        elif self.mock:
            self.model_caller = MockHJLModelCaller()
        else:
            api_key = os.environ.get("AGENT0_RESPONSES_API_KEY")
            if not api_key:
                raise ValueError(
                    "AGENT0_RESPONSES_API_KEY environment variable is required for live inference "
                    "when mock=False. Silent fallback to mock caller is forbidden."
                )
            self.model_caller = ResponsesHJLModelCaller(api_key=api_key)

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

    def _run_react_loop(
        self,
        image_path: str,
        instruction: str,
        category: str,
        max_steps: int | None,
        context: ToolExecutionContext,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], ReactDecisionResult | None]:
        """Internal ReAct loop executing against context with rollback and active image passing."""
        tool_cost = 0
        observations: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        limit = max_steps or self.config.max_steps
        reg = get_tool_registry()

        for step in range(1, limit + 1):
            current_image = _current_image_path(context)
            decision = self.model_caller.react_step(
                history=history,
                image_path=str(current_image),
                enabled_tools=self.config.enabled_tools,
                instruction=instruction,
                category=category,
            )

            if decision.action == ReactAction.FINISH:
                is_anom = decision.is_anomaly if decision.is_anomaly is not None else False
                result = {
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
                return result, history, decision

            # Tool call execution
            if decision.tool_name not in self.config.enabled_tools:
                obs_record = {
                    "step": step,
                    "tool": str(decision.tool_name),
                    "arguments": copy.deepcopy(decision.tool_arguments or {}),
                    "success": False,
                    "output_path": None,
                    "metadata": {},
                    "error": f"Tool '{decision.tool_name}' not allowed in enabled_tools.",
                    "retriable": False,
                }
                observations.append(obs_record)
                history.append({
                    "thought": f"Attempted unauthorized tool {decision.tool_name}",
                    "action": str(decision.tool_name),
                    "observation": obs_record,
                })
                continue

            checkpoint = context.checkpoint()
            tool_args = decision.tool_arguments or {}
            tool_result = execute_adapted_tool(
                name=decision.tool_name,
                arguments=tool_args,
                context=context,
                registry=reg,
            )
            tool_cost += 1

            if not tool_result.success:
                context.rollback(checkpoint)

            obs_record = {
                "step": step,
                "tool": decision.tool_name,
                "arguments": copy.deepcopy(tool_args),
                "success": tool_result.success,
                "output_path": tool_result.output_path,
                "metadata": copy.deepcopy(tool_result.metadata),
                "error": tool_result.error,
                "retriable": tool_result.retriable,
            }
            observations.append(obs_record)
            history.append({
                "thought": f"Executed tool {decision.tool_name}",
                "action": decision.tool_name,
                "observation": obs_record,
            })

        # Exceeded step limit without finishing: conclusion is UNRESOLVED and is_anomaly is None
        result = {
            "mode": "react",
            "image_path": image_path,
            "category": category,
            "is_anomaly": None,
            "conclusion": "UNRESOLVED",
            "confidence": 0.50,
            "total_steps": limit,
            "tool_cost": tool_cost,
            "observations": observations,
            "stop_reason": "MAX_STEPS",
        }
        return result, history, None

    def run_react(
        self,
        image_path: str,
        instruction: str = "Inspect this component for defects using tools.",
        category: str = "industrial_component",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Baseline 2: Standard ReAct loop with tools, without hierarchical checkpoints."""
        context = ToolExecutionContext(image=image_path)
        context["original_image_path"] = str(image_path)
        try:
            result, _, _ = self._run_react_loop(
                image_path=image_path,
                instruction=instruction,
                category=category,
                max_steps=max_steps,
                context=context,
            )
            return result
        finally:
            context.close()

    def run_react_verifier(
        self,
        image_path: str,
        instruction: str = "Inspect this component and verify the conclusion.",
        category: str = "industrial_component",
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Baseline 3: ReAct loop followed by generic VLM trajectory verification."""
        context = ToolExecutionContext(image=image_path)
        context["original_image_path"] = str(image_path)
        try:
            react_res, history, decision = self._run_react_loop(
                image_path=image_path,
                instruction=instruction,
                category=category,
                max_steps=max_steps,
                context=context,
            )

            # Verification runs while context is open and transformed images exist
            if decision is not None and decision.action == ReactAction.FINISH and decision.is_anomaly is not None:
                current_image = _current_image_path(context)
                verification = self.model_caller.generic_verify_react(
                    history=history,
                    decision=decision,
                    image_path=str(current_image),
                    category=category,
                )
                verifier_passed = verification.passed
                verifier_feedback = verification.feedback

                if not verifier_passed:
                    react_res["conclusion"] = "UNRESOLVED"
                    react_res["is_anomaly"] = None
            else:
                verifier_passed = None
                verifier_feedback = "Not applicable: ReAct produced no terminal claim."

            return {
                **react_res,
                "mode": "react_verifier",
                "verifier_passed": verifier_passed,
                "verifier_feedback": verifier_feedback,
            }
        finally:
            context.close()

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
        context["original_image_path"] = str(image_path)
        try:
            graph = create_hjl_graph(
                context=context,
                config=self.config,
                model_caller=self.model_caller,
                allow_synthetic=self.mock,
                live_mode=not self.mock,
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
