"""Explicit shared state and persistent evidence representation for HJL."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .taxonomy import (
    ActionType,
    CheckpointJudgment,
    EvidenceConclusion,
    EvidenceJudgment,
    EvidenceStatus,
    FailureType,
    GlobalStatus,
    RegionalStatus,
)


class HJLPhase(str, Enum):
    """Runtime phase of the Hierarchical Judgment Loop."""
    GLOBAL_DISCOVERY = "GLOBAL_DISCOVERY"          # Discovering candidate ROIs (active_hypothesis is None)
    HYPOTHESIS_INSPECTION = "HYPOTHESIS_INSPECTION" # Investigating active hypothesis on designated ROI
    EVIDENCE_RESOLUTION = "EVIDENCE_RESOLUTION"    # Resolving conflicts or confirming missing evidence after Evidence FAIL


class StopReason(str, Enum):
    """Explicit termination diagnosis for agent-level metrics."""
    CONFIRMED_ANOMALY = "CONFIRMED_ANOMALY"        # Evidence checkpoint passed with definitive ANOMALY
    CONFIRMED_NORMAL = "CONFIRMED_NORMAL"          # Global or Evidence checkpoint passed with definitive NORMAL
    MAX_STEPS = "MAX_STEPS"                        # Reached maximum allowed execution steps
    TOOL_FAILURE_LIMIT = "TOOL_FAILURE_LIMIT"      # Consecutive tool execution failures exceeded limit
    NO_VALID_ACTION = "NO_VALID_ACTION"            # Action space exhausted without remaining candidate actions


class EvidenceRelation(str, Enum):
    """Semantic relation of an observation to the active anomaly hypothesis."""
    SUPPORT = "SUPPORT"          # Confirms anomaly / active hypothesis
    CONTRADICT = "CONTRADICT"    # Directly refutes active hypothesis
    NEUTRAL = "NEUTRAL"          # Valid finding (clear ROI, normal reference match) but doesn't refute defects elsewhere


@dataclass
class EvidenceItem:
    """Atomic persistent observation item."""
    source_step: int
    region: list[int] | None          # [x1, y1, x2, y2]
    observation_type: str            # e.g., "texture_anomaly", "edge_crack", "reference_diff", "normal_feature"
    statement: str                   # Factual observation description
    relation: EvidenceRelation       # SUPPORT | CONTRADICT | NEUTRAL
    confidence: float                # Observation confidence [0.0, 1.0]
    source_tool: str                 # Tool that produced this observation
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_step": int(self.source_step),
            "region": list(self.region) if self.region is not None else None,
            "observation_type": str(self.observation_type),
            "statement": str(self.statement),
            "relation": self.relation.value if isinstance(self.relation, Enum) else str(self.relation),
            "confidence": float(self.confidence),
            "source_tool": str(self.source_tool),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceItem":
        rel = data.get("relation", "NEUTRAL")
        relation = EvidenceRelation(rel) if isinstance(rel, str) else rel
        return cls(
            source_step=int(data.get("source_step", 0)),
            region=list(data["region"]) if data.get("region") is not None else None,
            observation_type=str(data.get("observation_type", "general")),
            statement=str(data.get("statement", "")),
            relation=relation,
            confidence=float(data.get("confidence", 0.0)),
            source_tool=str(data.get("source_tool", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EvidenceState:
    """Persistent, structured evidence accumulator."""
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    supporting_evidence: list[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = field(default_factory=list)
    neutral_evidence: list[EvidenceItem] = field(default_factory=list)
    inspected_regions: list[list[int]] = field(default_factory=list)
    unresolved_regions: list[list[int]] = field(default_factory=list)
    normal_references: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    localization_confidence: float = 0.0
    tool_cost: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "supporting_evidence": [item.to_dict() for item in self.supporting_evidence],
            "contradicting_evidence": [item.to_dict() for item in self.contradicting_evidence],
            "neutral_evidence": [item.to_dict() for item in self.neutral_evidence],
            "inspected_regions": [list(r) for r in self.inspected_regions],
            "unresolved_regions": [list(r) for r in self.unresolved_regions],
            "normal_references": copy.deepcopy(self.normal_references),
            "hypotheses": copy.deepcopy(self.hypotheses),
            "unresolved_questions": list(self.unresolved_questions),
            "anomaly_score": float(self.anomaly_score),
            "localization_confidence": float(self.localization_confidence),
            "tool_cost": int(self.tool_cost),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceState":
        items = [EvidenceItem.from_dict(d) for d in data.get("evidence_items", [])]
        sup = [EvidenceItem.from_dict(d) for d in data.get("supporting_evidence", [])]
        contra = [EvidenceItem.from_dict(d) for d in data.get("contradicting_evidence", [])]
        neu = [EvidenceItem.from_dict(d) for d in data.get("neutral_evidence", [])]
        return cls(
            evidence_items=items,
            supporting_evidence=sup,
            contradicting_evidence=contra,
            neutral_evidence=neu,
            inspected_regions=[list(r) for r in data.get("inspected_regions", [])],
            unresolved_regions=[list(r) for r in data.get("unresolved_regions", [])],
            normal_references=list(data.get("normal_references", [])),
            hypotheses=list(data.get("hypotheses", [])),
            unresolved_questions=list(data.get("unresolved_questions", [])),
            anomaly_score=float(data.get("anomaly_score", 0.0)),
            localization_confidence=float(data.get("localization_confidence", 0.0)),
            tool_cost=int(data.get("tool_cost", 0)),
        )


@dataclass
class HJLState:
    """Explicit shared state passed between HJL graph nodes."""
    sample_id: str
    image_path: str
    instruction: str = "Determine whether the image has an industrial defect and describe it."
    category: str = "industrial_component"
    phase: HJLPhase = HJLPhase.HYPOTHESIS_INSPECTION

    global_observation: str | dict[str, Any] = ""
    candidate_regions: list[dict[str, Any]] = field(default_factory=list)

    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    active_hypothesis: dict[str, Any] | None = None

    current_plan: dict[str, Any] | None = None
    plan_history: list[dict[str, Any]] = field(default_factory=list)

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    evidence_state: EvidenceState = field(default_factory=EvidenceState)

    global_judgment: CheckpointJudgment | None = None
    regional_judgment: CheckpointJudgment | None = None
    evidence_judgment: EvidenceJudgment | None = None

    failure_type: FailureType | None = None
    failure_reason: str | None = None

    allowed_actions: list[ActionType] = field(default_factory=list)
    selected_action: ActionType | None = None

    current_step: int = 0
    max_steps: int = 8
    consecutive_tool_failures: int = 0
    stop_reason: StopReason | None = None

    final_prediction: dict[str, Any] | None = None
    step_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "instruction": self.instruction,
            "category": self.category,
            "phase": self.phase.value if isinstance(self.phase, Enum) else str(self.phase),
            "global_observation": copy.deepcopy(self.global_observation),
            "candidate_regions": copy.deepcopy(self.candidate_regions),
            "hypotheses": copy.deepcopy(self.hypotheses),
            "active_hypothesis": copy.deepcopy(self.active_hypothesis),
            "current_plan": copy.deepcopy(self.current_plan),
            "plan_history": copy.deepcopy(self.plan_history),
            "tool_calls": copy.deepcopy(self.tool_calls),
            "observations": copy.deepcopy(self.observations),
            "evidence_state": self.evidence_state.to_dict(),
            "global_judgment": self.global_judgment.to_dict() if self.global_judgment else None,
            "regional_judgment": self.regional_judgment.to_dict() if self.regional_judgment else None,
            "evidence_judgment": self.evidence_judgment.to_dict() if self.evidence_judgment else None,
            "failure_type": self.failure_type.value if isinstance(self.failure_type, Enum) else (str(self.failure_type) if self.failure_type else None),
            "failure_reason": self.failure_reason,
            "allowed_actions": [a.value if isinstance(a, Enum) else str(a) for a in self.allowed_actions],
            "selected_action": self.selected_action.value if isinstance(self.selected_action, Enum) else (str(self.selected_action) if self.selected_action else None),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "consecutive_tool_failures": self.consecutive_tool_failures,
            "stop_reason": self.stop_reason.value if isinstance(self.stop_reason, Enum) else (str(self.stop_reason) if self.stop_reason else None),
            "final_prediction": copy.deepcopy(self.final_prediction),
            "step_history": copy.deepcopy(self.step_history),
        }
