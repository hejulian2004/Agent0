"""Taxonomy definitions for HJL: Checkpoints, Judgments, Failure Types, and Actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class GlobalStatus(str, Enum):
    CONFIRMED_NORMAL = "CONFIRMED_NORMAL"
    PASS = "PASS"
    FAIL = "FAIL"


class RegionalStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class EvidenceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class EvidenceRelation(str, Enum):
    """Semantic relation of an observation to the active anomaly hypothesis."""
    SUPPORT = "SUPPORT"          # Confirms anomaly / active hypothesis
    CONTRADICT = "CONTRADICT"    # Directly refutes active hypothesis
    NEUTRAL = "NEUTRAL"          # Valid finding but doesn't refute defects elsewhere


class EvidenceConclusion(str, Enum):
    ANOMALY = "ANOMALY"
    NORMAL = "NORMAL"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class CheckpointJudgment:
    """Evaluation emitted by Global or Regional Verifiers."""
    status: GlobalStatus | RegionalStatus
    judgment_confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "judgment_confidence": float(self.judgment_confidence),
            "reason": str(self.reason),
        }


@dataclass
class EvidenceJudgment:
    """Definitive or unresolved judgment emitted by Evidence Verifier."""
    status: EvidenceStatus
    conclusion: EvidenceConclusion
    judgment_confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "conclusion": self.conclusion.value if isinstance(self.conclusion, Enum) else str(self.conclusion),
            "judgment_confidence": float(self.judgment_confidence),
            "reason": str(self.reason),
        }


class FailureType(str, Enum):
    """System-wide 8-class defect inspection failure taxonomy."""
    # Deterministic failure (tool crash / execution error)
    TOOL_FAILURE = "TOOL_FAILURE"

    # Semantic failure types (predicted by FailureDiagnoser)
    WRONG_REGION = "WRONG_REGION"
    LOW_RESOLUTION = "LOW_RESOLUTION"
    MISSING_REFERENCE = "MISSING_REFERENCE"
    VIEWPOINT_MISMATCH = "VIEWPOINT_MISMATCH"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    LOCALIZATION_UNCERTAIN = "LOCALIZATION_UNCERTAIN"
    PREMATURE_CONCLUSION = "PREMATURE_CONCLUSION"


SEMANTIC_FAILURE_TYPES = frozenset({
    FailureType.WRONG_REGION,
    FailureType.LOW_RESOLUTION,
    FailureType.MISSING_REFERENCE,
    FailureType.VIEWPOINT_MISMATCH,
    FailureType.CONTRADICTORY_EVIDENCE,
    FailureType.LOCALIZATION_UNCERTAIN,
    FailureType.PREMATURE_CONCLUSION,
})


class ActionType(str, Enum):
    """Abstract policy-level action primitives (Action Space Mask)."""
    GLOBAL_SCAN = "GLOBAL_SCAN"
    RELOCALIZE = "RELOCALIZE"
    ENHANCE_REGION = "ENHANCE_REGION"
    RETRIEVE_REFERENCE = "RETRIEVE_REFERENCE"
    ALIGN_VIEW = "ALIGN_VIEW"
    CROSS_VALIDATE = "CROSS_VALIDATE"
    RETRY_TOOL = "RETRY_TOOL"
    INSPECT_NEXT_REGION = "INSPECT_NEXT_REGION"


@dataclass
class FailureDiagnosis:
    """Attribution and root-cause explanation for an inspection failure."""
    failure_type: FailureType
    cause: str
    diagnosis_confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value if isinstance(self.failure_type, Enum) else str(self.failure_type),
            "cause": str(self.cause),
            "diagnosis_confidence": float(self.diagnosis_confidence),
        }
