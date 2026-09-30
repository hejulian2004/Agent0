"""Strictly typed dataclasses, invariants, and validation helpers for HJL."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .taxonomy import (
    EvidenceConclusion,
    EvidenceRelation,
    EvidenceStatus,
    FailureType,
    RegionalStatus,
)


class ModelOutputError(RuntimeError):
    """Raised when model response cannot be parsed or violates schema constraints."""


class ReactAction(str, Enum):
    TOOL_CALL = "TOOL_CALL"
    FINISH = "FINISH"


def require_bool(data: Mapping[str, Any], key: str) -> bool:
    """Validate that a required key exists and is strictly of type bool."""
    if key not in data:
        raise ValueError(f"Missing required boolean field: '{key}'")
    val = data[key]
    if type(val) is not bool:
        raise ValueError(f"Field '{key}' must be boolean, got {type(val).__name__}: {val!r}")
    return val


def require_float(
    data: Mapping[str, Any],
    key: str,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """Validate that a required key exists and is a valid float within [min_val, max_val]."""
    if key not in data:
        raise ValueError(f"Missing required numeric field: '{key}'")
    val = data[key]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Field '{key}' must be numeric, got {type(val).__name__}: {val!r}")
    num = float(val)
    if not (min_val <= num <= max_val):
        raise ValueError(f"Field '{key}' must be in [{min_val}, {max_val}], got {num}")
    return num


def require_str(data: Mapping[str, Any], key: str, min_len: int = 1) -> str:
    """Validate that a required key exists and is a non-empty string."""
    if key not in data:
        raise ValueError(f"Missing required string field: '{key}'")
    val = data[key]
    if not isinstance(val, str) or len(val.strip()) < min_len:
        raise ValueError(f"Field '{key}' must be a non-empty string, got {val!r}")
    return val.strip()


def require_dict(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Validate that a required key exists and is a dictionary."""
    if key not in data:
        raise ValueError(f"Missing required dict field: '{key}'")
    val = data[key]
    if not isinstance(val, dict):
        raise ValueError(f"Field '{key}' must be a dict, got {type(val).__name__}")
    return dict(val)


def require_list(data: Mapping[str, Any], key: str) -> list[Any]:
    """Validate that a required key exists and is a list."""
    if key not in data:
        raise ValueError(f"Missing required list field: '{key}'")
    val = data[key]
    if not isinstance(val, list):
        raise ValueError(f"Field '{key}' must be a list, got {type(val).__name__}")
    return list(val)


@dataclass
class CandidateRegion:
    bbox: list[int]
    confidence: float
    label: str | None = None
    region_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have length 4, got {self.bbox}")
        if self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
            raise ValueError(f"Invalid bbox dimensions [x1,y1,x2,y2]: {self.bbox}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def validate_bounds(self, width: int, height: int) -> None:
        """Validate bounding box falls within image boundaries."""
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(f"bbox {self.bbox} exceeds image dimensions [{width}x{height}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "confidence": float(self.confidence),
            "label": self.label,
            "region_id": self.region_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateRegion":
        return cls(
            bbox=list(data["bbox"]),
            confidence=float(data["confidence"]),
            label=data.get("label"),
            region_id=data.get("region_id"),
        )


@dataclass
class GlobalInspectionResult:
    observation: str
    candidate_regions: list[CandidateRegion]
    is_normal: bool
    confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "candidate_regions": [c.to_dict() for c in self.candidate_regions],
            "is_normal": self.is_normal,
            "confidence": self.confidence,
        }


@dataclass
class HypothesisResult:
    hypothesis_id: str
    type: str
    description: str
    confidence: float
    target_region: list[int] | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "type": self.type,
            "description": self.description,
            "confidence": self.confidence,
            "target_region": list(self.target_region) if self.target_region else None,
        }


@dataclass
class RegionalVerificationResult:
    status: RegionalStatus
    judgment_confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.judgment_confidence <= 1.0):
            raise ValueError(f"judgment_confidence must be in [0, 1], got {self.judgment_confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "judgment_confidence": self.judgment_confidence,
            "reason": self.reason,
        }


@dataclass
class RegionalEvidenceFinding:
    finding: str
    relation: EvidenceRelation
    observation_type: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding": self.finding,
            "relation": self.relation.value if isinstance(self.relation, Enum) else str(self.relation),
            "observation_type": self.observation_type,
            "confidence": self.confidence,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class EvidenceVerificationResult:
    status: EvidenceStatus
    conclusion: EvidenceConclusion
    judgment_confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.judgment_confidence <= 1.0):
            raise ValueError(f"judgment_confidence must be in [0, 1], got {self.judgment_confidence}")
        if self.status == EvidenceStatus.PASS and self.conclusion == EvidenceConclusion.UNRESOLVED:
            raise ValueError("EvidenceStatus.PASS cannot have conclusion UNRESOLVED")
        if self.status == EvidenceStatus.FAIL and self.conclusion != EvidenceConclusion.UNRESOLVED:
            raise ValueError(f"EvidenceStatus.FAIL must have conclusion UNRESOLVED, got {self.conclusion}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "conclusion": self.conclusion.value if isinstance(self.conclusion, Enum) else str(self.conclusion),
            "judgment_confidence": self.judgment_confidence,
            "reason": self.reason,
        }


@dataclass
class FailureDiagnosisResult:
    failure_type: FailureType
    cause: str
    diagnosis_confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.diagnosis_confidence <= 1.0):
            raise ValueError(f"diagnosis_confidence must be in [0, 1], got {self.diagnosis_confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value if isinstance(self.failure_type, Enum) else str(self.failure_type),
            "cause": self.cause,
            "diagnosis_confidence": self.diagnosis_confidence,
        }


@dataclass
class GenericVerificationResult:
    passed: bool
    confidence: float
    feedback: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "confidence": float(self.confidence),
            "feedback": str(self.feedback),
        }


@dataclass
class DirectPredictionResult:
    is_anomaly: bool
    conclusion: str
    anomaly_score: float
    confidence: float
    explanation: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.anomaly_score <= 1.0):
            raise ValueError(f"anomaly_score must be in [0, 1], got {self.anomaly_score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.conclusion == "ANOMALY" and not self.is_anomaly:
            raise ValueError("conclusion 'ANOMALY' requires is_anomaly=True")
        if self.conclusion == "NORMAL" and self.is_anomaly:
            raise ValueError("conclusion 'NORMAL' requires is_anomaly=False")
        if self.is_anomaly and self.conclusion != "ANOMALY":
            raise ValueError("is_anomaly=True requires conclusion 'ANOMALY'")
        if not self.is_anomaly and self.conclusion != "NORMAL":
            raise ValueError("is_anomaly=False requires conclusion 'NORMAL'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "conclusion": self.conclusion,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class ReactDecisionResult:
    action: ReactAction
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    final_answer: str | None = None
    is_anomaly: bool | None = None
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.action == ReactAction.TOOL_CALL:
            if not self.tool_name:
                raise ValueError("ReactAction.TOOL_CALL requires tool_name")
            if self.tool_arguments is None or not isinstance(self.tool_arguments, dict):
                raise ValueError("ReactAction.TOOL_CALL requires tool_arguments as a dict")
            if self.is_anomaly is not None:
                raise ValueError("ReactAction.TOOL_CALL must have is_anomaly=None")
        elif self.action == ReactAction.FINISH:
            if self.tool_name is not None:
                raise ValueError("ReactAction.FINISH must not specify tool_name")
            if self.tool_arguments is not None:
                raise ValueError("ReactAction.FINISH must not specify tool_arguments")
            if self.is_anomaly is None or type(self.is_anomaly) is not bool:
                raise ValueError("ReactAction.FINISH requires boolean is_anomaly")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value if isinstance(self.action, Enum) else str(self.action),
            "tool_name": self.tool_name,
            "tool_arguments": copy.deepcopy(self.tool_arguments) if self.tool_arguments is not None else None,
            "final_answer": self.final_answer,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
        }
