"""Source-compatible local SFT construction infrastructure.

This package is deliberately additive: it supplements the upstream Agent0-VL
runtime without changing any upstream prompt, evaluator, sandbox, or trainer
file.  The internal audit schema is richer than the final exported JSONL;
exported rows are restricted to the upstream Swift ``messages + images``
shape.
"""

from .schema import (
    AuditRecord,
    ExportCandidate,
    ExportRow,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SupervisionUnit,
    TrajectoryRecord,
    ValidationDecision,
    migrate_v1_to_v2,
)
from .answer_check import (
    ANSWER_CHECK_METHOD,
    EVIDENCE_SOURCE,
    check_reference_answer,
    extract_final_answer,
    normalize_answer,
)
from .budget import TeacherBudgetExceeded, TeacherRequestBudget
from .dedup import exact_dedup_key, exact_deduplicate
from .projector import Projector
from .snapshot_store import SnapshotStore, SnapshotStoreError
from .snapshots import ImmutableSnapshot
from .teacher_backend import SamplingConfig, TeacherBackend, TeacherResponse
from .trajectory_builder import RealTrajectoryBuilder, RolloutOutcome
from .validator import validate_supervision_unit, validate_trajectory_for_export
from .pipeline import PipelineResult, export_pipeline_result, validate_project_deduplicate
from .source_adapter import (
    DuplicateSourceIdentityError,
    ForbiddenIndexError,
    ForbiddenSourceIndex,
    InvalidImageRootError,
    RealSourceAdapter,
    SourceInputError,
    SourcePreflightEntry,
    SourcePreflightError,
    SourcePreflightResult,
)

__all__ = [
    "AuditRecord",
    "ExportCandidate",
    "ExportRow",
    "LEGACY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SupervisionUnit",
    "TrajectoryRecord",
    "ValidationDecision",
    "migrate_v1_to_v2",
    "ANSWER_CHECK_METHOD",
    "EVIDENCE_SOURCE",
    "check_reference_answer",
    "extract_final_answer",
    "normalize_answer",
    "TeacherBudgetExceeded",
    "TeacherRequestBudget",
    "exact_dedup_key",
    "exact_deduplicate",
    "Projector",
    "SnapshotStore",
    "SnapshotStoreError",
    "ImmutableSnapshot",
    "SamplingConfig",
    "TeacherBackend",
    "TeacherResponse",
    "RealTrajectoryBuilder",
    "RolloutOutcome",
    "validate_supervision_unit",
    "validate_trajectory_for_export",
    "PipelineResult",
    "export_pipeline_result",
    "validate_project_deduplicate",
    "DuplicateSourceIdentityError",
    "ForbiddenIndexError",
    "ForbiddenSourceIndex",
    "InvalidImageRootError",
    "RealSourceAdapter",
    "SourceInputError",
    "SourcePreflightEntry",
    "SourcePreflightError",
    "SourcePreflightResult",
]
