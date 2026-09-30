"""Lightweight, deterministic StateGraph implementation for HJL with tool-type routing."""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

from agent0_protocol.tools import ToolExecutionContext

from .config import HJLConfig
from .model_caller import HJLModelCaller
from .state import HJLPhase, HJLState, StopReason
from .taxonomy import EvidenceStatus, FailureType, GlobalStatus, RegionalStatus

logger = logging.getLogger(__name__)

START = "__start__"
END = "__end__"

NodeFn = Callable[[HJLState], HJLState | dict[str, Any]]
RouterFn = Callable[[HJLState], str]


class CompiledGraph:
    """Compiled state graph ready for deterministic stepwise execution."""

    def __init__(
        self,
        nodes: dict[str, NodeFn],
        edges: dict[str, str],
        conditional_edges: dict[str, tuple[RouterFn, dict[str, str] | None]],
        entry_point: str,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point

    def run(self, initial_state: HJLState, max_graph_transitions: int = 100) -> HJLState:
        """Execute state through graph until END or a terminal condition is reached."""
        state = initial_state
        current_node = self.entry_point
        transitions = 0

        while current_node != END and transitions < max_graph_transitions:
            transitions += 1

            if current_node not in self.nodes:
                raise RuntimeError(f"Unknown graph node: {current_node!r}")

            # Hard step budget check across all transitions
            if state.current_step >= state.max_steps and state.stop_reason is None and current_node != "finalizer":
                state.stop_reason = StopReason.MAX_STEPS
                current_node = "finalizer"

            node_fn = self.nodes[current_node]
            logger.debug(f"[HJL Graph] Executing node: {current_node} (step {state.current_step})")

            # Execute node function
            result = node_fn(state)
            if isinstance(result, HJLState):
                state = result
            elif isinstance(result, dict):
                for k, v in result.items():
                    if hasattr(state, k):
                        setattr(state, k, v)

            # Record granular step transition snapshot into step_history
            snapshot = {
                "transition": transitions,
                "node": current_node,
                "step": state.current_step,
                "phase": state.phase.value if hasattr(state.phase, "value") else str(state.phase),
                "hypothesis": copy.deepcopy(state.active_hypothesis),
                "allowed_actions": [a.value if hasattr(a, "value") else str(a) for a in state.allowed_actions],
                "selected_action": state.selected_action.value if hasattr(state.selected_action, "value") else (str(state.selected_action) if state.selected_action else None),
                "tool_call": copy.deepcopy(state.tool_calls[0]) if state.tool_calls else None,
                "latest_observation": copy.deepcopy(state.observations[-1]) if state.observations else None,
                "regional_judgment": state.regional_judgment.to_dict() if state.regional_judgment else None,
                "failure_type": state.failure_type.value if hasattr(state.failure_type, "value") else (str(state.failure_type) if state.failure_type else None),
                "evidence_judgment": state.evidence_judgment.to_dict() if state.evidence_judgment else None,
                "anomaly_score": round(state.evidence_state.anomaly_score, 4),
                "stop_reason": state.stop_reason.value if hasattr(state.stop_reason, "value") else (str(state.stop_reason) if state.stop_reason else None),
            }
            state.step_history.append(snapshot)

            # Check if state reached a terminal stop reason at finalizer
            if current_node == "finalizer":
                break

            # Centralized terminal intercept for any non-None stop_reason
            if state.stop_reason is not None:
                current_node = "finalizer"
                continue

            # Determine next node
            if current_node in self.conditional_edges:
                router_fn, path_map = self.conditional_edges[current_node]
                route_key = router_fn(state)
                if path_map is not None:
                    next_node = path_map.get(route_key, route_key)
                else:
                    next_node = route_key
            elif current_node in self.edges:
                next_node = self.edges[current_node]
            else:
                next_node = END

            current_node = next_node

        if transitions >= max_graph_transitions and state.stop_reason is None:
            state.stop_reason = StopReason.MAX_STEPS

        return state


class StateGraph:
    """Builder for constructing deterministic HJL state machines."""

    def __init__(self, state_schema: type = HJLState) -> None:
        self.state_schema = state_schema
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._conditional_edges: dict[str, tuple[RouterFn, dict[str, str] | None]] = {}
        self._entry_point: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in self._nodes:
            raise ValueError(f"Duplicate node name: {name}")
        self._nodes[name] = fn
        return self

    def add_edge(self, from_node: str, to_node: str) -> "StateGraph":
        if from_node == START:
            self._entry_point = to_node
        else:
            self._edges[from_node] = to_node
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        router_fn: RouterFn,
        path_map: dict[str, str] | None = None,
    ) -> "StateGraph":
        self._conditional_edges[from_node] = (router_fn, path_map)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self._entry_point = name
        return self

    def compile(self) -> CompiledGraph:
        if not self._entry_point:
            raise ValueError("StateGraph entry point is not set.")
        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional_edges=dict(self._conditional_edges),
            entry_point=self._entry_point,
        )


def create_hjl_graph(
    context: ToolExecutionContext | None = None,
    config: HJLConfig | None = None,
    model_caller: HJLModelCaller | None = None,
    allow_synthetic: bool = False,
    live_mode: bool = False,
) -> CompiledGraph:
    """Build and compile the canonical Hierarchical Judgment Loop graph with tool-type routing."""
    from .nodes.evidence_extractor import evidence_extractor_node
    from .nodes.evidence_updater import evidence_updater_node
    from .nodes.evidence_verifier import evidence_verifier_node
    from .nodes.failure_diagnoser import failure_diagnoser_node
    from .nodes.finalizer import finalizer_node
    from .nodes.global_inspector import global_inspector_node
    from .nodes.global_verifier import global_verifier_node
    from .nodes.hypothesis_generator import hypothesis_generator_node
    from .nodes.planner import planner_node
    from .nodes.regional_verifier import regional_verifier_node
    from .nodes.replanner import replanner_node
    from .nodes.state_updaters import (
        candidate_state_updater_node,
        comparison_evidence_extractor_node,
        reference_state_updater_node,
    )
    from .nodes.tool_executor import tool_executor_node

    cfg = config or HJLConfig()

    workflow = StateGraph(state_schema=HJLState)

    # Wrap nodes with bound dependencies
    workflow.add_node(
        "global_inspector",
        lambda s: global_inspector_node(s, model_caller=model_caller, live_mode=live_mode),
    )
    workflow.add_node(
        "hypothesis_generator",
        lambda s: hypothesis_generator_node(s, model_caller=model_caller, live_mode=live_mode),
    )
    workflow.add_node(
        "global_verifier",
        lambda s: global_verifier_node(
            s, global_normal_confidence_threshold=cfg.global_normal_confidence_threshold
        ),
    )
    workflow.add_node(
        "planner",
        lambda s: planner_node(
            s,
            reference_corpus_dir=cfg.reference_corpus_dir,
            allow_synthetic=allow_synthetic,
        ),
    )
    workflow.add_node("tool_executor", lambda s: tool_executor_node(s, context=context))
    workflow.add_node("regional_verifier", regional_verifier_node)
    workflow.add_node(
        "evidence_extractor",
        lambda s: evidence_extractor_node(s, model_caller=model_caller, live_mode=live_mode),
    )
    workflow.add_node("reference_state_updater", reference_state_updater_node)
    workflow.add_node(
        "comparison_evidence_extractor",
        lambda s: comparison_evidence_extractor_node(
            s, reference_similarity_threshold=cfg.reference_similarity_threshold
        ),
    )
    workflow.add_node(
        "candidate_state_updater",
        lambda s: candidate_state_updater_node(s, max_discovery_attempts=cfg.max_discovery_attempts),
    )
    workflow.add_node(
        "evidence_updater",
        lambda s: evidence_updater_node(s, reference_similarity_threshold=cfg.reference_similarity_threshold),
    )
    workflow.add_node(
        "evidence_verifier",
        lambda s: evidence_verifier_node(
            s,
            anomaly_threshold=cfg.anomaly_threshold,
            normal_threshold=cfg.normal_threshold,
            checkpoint_confidence_threshold=cfg.checkpoint_confidence_threshold,
            min_evidence_count=cfg.min_evidence_count,
        ),
    )
    workflow.add_node(
        "failure_diagnoser",
        lambda s: failure_diagnoser_node(s, model_caller=model_caller, live_mode=live_mode),
    )
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("finalizer", finalizer_node)

    workflow.add_edge(START, "global_inspector")
    workflow.add_edge("global_inspector", "global_verifier")

    def route_global(state: HJLState) -> str:
        if state.global_judgment and state.global_judgment.status == GlobalStatus.CONFIRMED_NORMAL:
            return "finalizer"
        elif state.global_judgment and state.global_judgment.status == GlobalStatus.FAIL:
            return "planner"  # in GLOBAL_DISCOVERY phase
        return "hypothesis_generator"

    workflow.add_conditional_edges("global_verifier", route_global)
    workflow.add_edge("hypothesis_generator", "planner")
    workflow.add_edge("planner", "tool_executor")

    # Tool-Type-Specific Routing
    def route_tool_executor(state: HJLState) -> str:
        if state.failure_type == FailureType.TOOL_FAILURE:
            return "replanner"
        if not state.observations:
            return "replanner"

        latest_tool = state.observations[-1].get("tool", "")
        if latest_tool in {"crop_region", "zoom_region", "rotate_image"}:
            return "regional_verifier"
        elif latest_tool == "retrieve_normal_reference":
            return "reference_state_updater"
        elif latest_tool == "compare_with_reference":
            return "comparison_evidence_extractor"
        elif latest_tool == "localize_candidate":
            return "candidate_state_updater"
        return "regional_verifier"

    workflow.add_conditional_edges("tool_executor", route_tool_executor)

    # Reference retrieval branch -> Planner (CROSS_VALIDATE)
    workflow.add_edge("reference_state_updater", "planner")

    # Reference comparison branch -> Evidence Updater -> Evidence Verifier
    workflow.add_edge("comparison_evidence_extractor", "evidence_updater")

    # Candidate localization branch -> Hypothesis Generator or Planner
    def route_candidate_updater(state: HJLState) -> str:
        if state.stop_reason is not None:
            return "finalizer"
        if state.candidate_regions:
            return "hypothesis_generator"
        return "planner"

    workflow.add_conditional_edges("candidate_state_updater", route_candidate_updater)

    # Spatial ROI branch: Regional Verifier -> Evidence Extractor (PASS) or Failure Diagnoser (FAIL)
    def route_regional(state: HJLState) -> str:
        if state.regional_judgment and state.regional_judgment.status == RegionalStatus.PASS:
            return "evidence_extractor"
        return "failure_diagnoser"

    workflow.add_conditional_edges("regional_verifier", route_regional)
    workflow.add_edge("evidence_extractor", "evidence_updater")
    workflow.add_edge("evidence_updater", "evidence_verifier")

    # Evidence Verifier -> Finalizer (PASS) or Failure Diagnoser (FAIL)
    def route_evidence(state: HJLState) -> str:
        if state.evidence_judgment and state.evidence_judgment.status == EvidenceStatus.PASS:
            return "finalizer"
        return "failure_diagnoser"

    workflow.add_conditional_edges("evidence_verifier", route_evidence)
    workflow.add_edge("failure_diagnoser", "replanner")

    def route_replanner(state: HJLState) -> str:
        if state.stop_reason is not None:
            return "finalizer"
        return "planner"

    workflow.add_conditional_edges("replanner", route_replanner)
    workflow.add_edge("finalizer", END)

    return workflow.compile()
