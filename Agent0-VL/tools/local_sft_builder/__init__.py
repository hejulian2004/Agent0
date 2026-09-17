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
from .budget import TeacherBudgetExceeded, TeacherRequestBudget
from .dedup import exact_dedup_key, exact_deduplicate
from .projector import Projector
from .snapshots import ImmutableSnapshot
from .validator import validate_supervision_unit, validate_trajectory_for_export
from .pipeline import PipelineResult, export_pipeline_result, validate_project_deduplicate

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
    "TeacherBudgetExceeded",
    "TeacherRequestBudget",
    "exact_dedup_key",
    "exact_deduplicate",
    "Projector",
    "ImmutableSnapshot",
    "validate_supervision_unit",
    "validate_trajectory_for_export",
    "PipelineResult",
    "export_pipeline_result",
    "validate_project_deduplicate",
]
