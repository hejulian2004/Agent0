"""Internal audit and export models for the local SFT builder.

The audit models intentionally retain provenance and validation information.
The :class:`ExportRow` serializer intentionally emits only the two fields
accepted by the upstream Swift SFT entry points: ``messages`` and ``images``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = "agent0vl.dataset.v2"
LEGACY_SCHEMA_VERSION = "agent0vl.dataset.v1"
DEDUP_KEY_VERSION = "sft_exact_v1"
SNAPSHOT_SCHEMA_VERSION = "agent0vl.snapshot.v1"

SUPERVISION_TYPES = frozenset(
    {"solver_positive", "verifier", "repair", "regeneration"}
)
SUPERVISION_STATUSES = frozenset(
    {"accepted", "audit_only", "review_required", "rejected"}
)


class MigrationError(ValueError):
    """Base class for explicit schema migration failures."""


class MigrationConflictError(MigrationError):
    """Raised when legacy and v2 supervision fields disagree."""


def migrate_v1_to_v2(record: Mapping[str, Any]) -> dict[str, Any]:
    """Pure, idempotent migration from the legacy supervision field.

    Rules:
    * the input mapping is never mutated;
    * ``record_type`` maps to ``supervision_type``;
    * equal duplicate fields are accepted but the legacy field is removed;
    * conflicting fields raise :class:`MigrationConflictError`;
    * ``migrated_from_schema_version`` is written only when a legacy record was
      actually normalized or upgraded.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    result = copy.deepcopy(dict(record))
    has_legacy_field = "record_type" in result
    has_v2_field = "supervision_type" in result
    legacy_value = result.get("record_type")
    v2_value = result.get("supervision_type")

    if has_legacy_field and has_v2_field and legacy_value != v2_value:
        raise MigrationConflictError(
            "record_type and supervision_type conflict: "
            f"{legacy_value!r} != {v2_value!r}"
        )

    schema_version = result.get("schema_version")
    actual_migration = False

    if has_legacy_field:
        if not has_v2_field:
            result["supervision_type"] = legacy_value
        result.pop("record_type", None)
        actual_migration = schema_version != SCHEMA_VERSION

    if schema_version == LEGACY_SCHEMA_VERSION:
        result["schema_version"] = SCHEMA_VERSION
        actual_migration = True
    elif schema_version is None:
        # A legacy field is sufficient evidence that this is a v1-shaped row.
        # A modern field by itself is treated as an already-v2 row.
        if has_legacy_field:
            result["schema_version"] = SCHEMA_VERSION
            actual_migration = True
        else:
            result["schema_version"] = SCHEMA_VERSION

    supervision_type = result.get("supervision_type")
    if supervision_type is not None and supervision_type not in SUPERVISION_TYPES:
        raise MigrationError(f"unknown supervision_type: {supervision_type!r}")

    if actual_migration and "migrated_from_schema_version" not in result:
        result["migrated_from_schema_version"] = LEGACY_SCHEMA_VERSION

    return result


def _copy_messages(messages: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(copy.deepcopy(list(messages)))


@dataclass(frozen=True)
class ValidationDecision:
    """Validator output consumed by the Projector."""

    status: str
    supervision_type: str | None
    reasons: tuple[str, ...] = ()
    evidence_sources: tuple[str, ...] = ()
    solver_positive_eligible: bool = False
    exportable: bool = False

    def __post_init__(self) -> None:
        if self.status not in SUPERVISION_STATUSES:
            raise ValueError(f"unknown supervision status: {self.status}")
        if self.supervision_type is not None and self.supervision_type not in SUPERVISION_TYPES:
            raise ValueError(f"unknown supervision type: {self.supervision_type}")
        if self.exportable and not (
            self.status == "accepted" and self.supervision_type == "solver_positive"
        ):
            raise ValueError("only accepted solver_positive decisions are exportable")

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "supervision_type": self.supervision_type,
            "reasons": list(self.reasons),
            "evidence_sources": list(self.evidence_sources),
            "solver_positive_eligible": self.solver_positive_eligible,
            "exportable": self.exportable,
        }


@dataclass(frozen=True)
class SupervisionUnit:
    """One internal target with exactly one assistant target and role."""

    record_id: str
    task_id: str
    trajectory_id: str
    supervision_type: str
    context_messages: tuple[dict[str, Any], ...]
    assistant_target: str
    supervision_status: str = "accepted"
    evidence_sources: tuple[str, ...] = ()
    controlled_ancestry: bool = False
    repaired_error: bool = False
    regenerated: bool = False
    step_index: int | None = None

    def __post_init__(self) -> None:
        if self.supervision_type not in SUPERVISION_TYPES:
            raise ValueError(f"unknown supervision type: {self.supervision_type}")
        if self.supervision_status not in SUPERVISION_STATUSES:
            raise ValueError(f"unknown supervision status: {self.supervision_status}")
        if not self.assistant_target:
            raise ValueError("assistant_target must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_id": self.record_id,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "supervision_type": self.supervision_type,
            "context_messages": copy.deepcopy(list(self.context_messages)),
            "assistant_target": self.assistant_target,
            "supervision_status": self.supervision_status,
            "evidence_sources": list(self.evidence_sources),
            "controlled_ancestry": self.controlled_ancestry,
            "repaired_error": self.repaired_error,
            "regenerated": self.regenerated,
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class AuditRecord:
    """Generic append-only audit envelope."""

    record_id: str
    task_id: str
    record_kind: str
    payload: Mapping[str, Any]
    supervision_status: str = "audit_only"

    def __post_init__(self) -> None:
        if self.supervision_status not in SUPERVISION_STATUSES:
            raise ValueError(f"unknown supervision status: {self.supervision_status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_id": self.record_id,
            "task_id": self.task_id,
            "record_kind": self.record_kind,
            "payload": copy.deepcopy(dict(self.payload)),
            "supervision_status": self.supervision_status,
        }


@dataclass(frozen=True)
class TrajectoryRecord:
    """A complete trajectory plus immutable lineage facts."""

    task_id: str
    trajectory_id: str
    source_record_id: str
    stage: str
    messages: tuple[dict[str, Any], ...]
    images: tuple[str, ...] = ()
    image_content_hashes: tuple[str, ...] = ()
    canonical_source: str = "natural"
    lineage_events: tuple[str, ...] = ()
    ancestor_trajectory_ids: tuple[str, ...] = ()
    root_snapshot_hash: str | None = None
    start_snapshot_hash: str | None = None
    controlled_intervention: bool = False
    audit_context_received: bool = False
    final_answer_valid: bool = False
    step_validated: tuple[bool, ...] = ()
    repaired_error_steps: tuple[int, ...] = ()
    regenerated_steps: tuple[int, ...] = ()
    target_repair_depth: int = 0
    target_repair_depth_frozen: int = 0
    canonical_actual_repair_count: int = 0
    branch_actual_repair_count: int = 0
    supervision_units: tuple[SupervisionUnit, ...] = ()

    @property
    def has_controlled_ancestry(self) -> bool:
        controlled_markers = {
            "controlled",
            "verifier",
            "repair",
            "regeneration",
            "controlled_intervention",
        }
        return bool(
            self.controlled_intervention
            or self.audit_context_received
            or controlled_markers.intersection(self.lineage_events)
            or any(unit.controlled_ancestry for unit in self.supervision_units)
        )

    @property
    def clean_natural_lineage(self) -> bool:
        return (
            self.canonical_source in {"natural", "natural_replay"}
            and not self.has_controlled_ancestry
            and not self.repaired_error_steps
            and not self.regenerated_steps
            and self.root_snapshot_hash is not None
            and self.start_snapshot_hash == self.root_snapshot_hash
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "source_record_id": self.source_record_id,
            "stage": self.stage,
            "messages": copy.deepcopy(list(self.messages)),
            "images": list(self.images),
            "image_content_hashes": list(self.image_content_hashes),
            "canonical_source": self.canonical_source,
            "lineage_events": list(self.lineage_events),
            "ancestor_trajectory_ids": list(self.ancestor_trajectory_ids),
            "root_snapshot_hash": self.root_snapshot_hash,
            "start_snapshot_hash": self.start_snapshot_hash,
            "controlled_intervention": self.controlled_intervention,
            "audit_context_received": self.audit_context_received,
            "final_answer_valid": self.final_answer_valid,
            "step_validated": list(self.step_validated),
            "repaired_error_steps": list(self.repaired_error_steps),
            "regenerated_steps": list(self.regenerated_steps),
            "target_repair_depth": self.target_repair_depth,
            "target_repair_depth_frozen": self.target_repair_depth_frozen,
            "canonical_actual_repair_count": self.canonical_actual_repair_count,
            "branch_actual_repair_count": self.branch_actual_repair_count,
            "supervision_units": [unit.to_dict() for unit in self.supervision_units],
        }


@dataclass(frozen=True)
class ExportCandidate:
    """Validator-approved complete clean Solver conversation."""

    task_id: str
    trajectory_id: str
    source_record_id: str
    stage: str
    messages: tuple[dict[str, Any], ...]
    images: tuple[str, ...]
    image_content_hashes: tuple[str, ...]
    validation: ValidationDecision


@dataclass(frozen=True)
class ExportRow:
    """Final Swift row; metadata is retained only in memory for dedup ordering."""

    messages: tuple[dict[str, Any], ...]
    images: tuple[str, ...]
    image_content_hashes: tuple[str, ...] = ()
    source_record_id: str = ""
    trajectory_id: str = ""
    stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        # Deliberately no provenance, schema, status, or supervision fields.
        return {
            "messages": copy.deepcopy(list(self.messages)),
            "images": list(self.images),
        }
