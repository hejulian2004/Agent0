"""Lightweight, deterministic StateGraph implementation for HJL.

Mirrors LangGraph's API signatures (START, END, add_node, add_edge, add_conditional_edges, run)
with zero external unpinned dependencies.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

from .state import HJLState, StopReason

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
        """Execute state through graph until END or a StopReason is reached."""
        state = initial_state
        current_node = self.entry_point
        transitions = 0

        while current_node != END and transitions < max_graph_transitions:
            transitions += 1

            if current_node not in self.nodes:
                raise RuntimeError(f"Unknown graph node: {current_node!r}")

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

            # Check if state reached a terminal stop reason at finalizer
            if current_node == "finalizer":
                break

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


def create_hjl_graph() -> CompiledGraph:
    """Build and compile the canonical Hierarchical Judgment Loop graph."""
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
    from .nodes.tool_executor import tool_executor_node
    from .taxonomy import EvidenceStatus, FailureType, GlobalStatus, RegionalStatus

    workflow = StateGraph(state_schema=HJLState)

    workflow.add_node("global_inspector", global_inspector_node)
    workflow.add_node("hypothesis_generator", hypothesis_generator_node)
    workflow.add_node("global_verifier", global_verifier_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("regional_verifier", regional_verifier_node)
    workflow.add_node("evidence_updater", evidence_updater_node)
    workflow.add_node("evidence_verifier", evidence_verifier_node)
    workflow.add_node("failure_diagnoser", failure_diagnoser_node)
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

    def route_tool_executor(state: HJLState) -> str:
        if state.failure_type == FailureType.TOOL_FAILURE:
            return "replanner"
        return "regional_verifier"

    workflow.add_conditional_edges("tool_executor", route_tool_executor)

    def route_regional(state: HJLState) -> str:
        if state.regional_judgment and state.regional_judgment.status == RegionalStatus.PASS:
            return "evidence_updater"
        return "failure_diagnoser"

    workflow.add_conditional_edges("regional_verifier", route_regional)
    workflow.add_edge("evidence_updater", "evidence_verifier")

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
