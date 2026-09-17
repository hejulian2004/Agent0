"""Validation policy for audit units and clean Solver export candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .protocol import ProtocolError, validate_solver_final
from .schema import (
    ExportCandidate,
    SUPERVISION_TYPES,
    SupervisionUnit,
    TrajectoryRecord,
    ValidationDecision,
)
from .source_guard import SourceGuardDecision


LOCAL_EVIDENCE_SOURCES = frozenset(
    {
        "sandbox",
        "symbolic_recomputation",
        "numeric_recomputation",
        "image_evidence",
        "deterministic_rule",
        "independent_checker",
    }
)
CONTROLLED_EVENTS = frozenset(
    {"controlled", "verifier", "repair", "regeneration", "controlled_intervention"}
)


@dataclass(frozen=True)
class RepairDepthStats:
    target_repair_depth: int
    canonical_actual_repair_count: int
    branch_actual_repair_count: int


def _message_shape_valid(messages: Iterable[dict[str, Any]]) -> tuple[bool, str | None]:
    items = list(messages)
    if not items:
        return False, "empty_messages"
    if any(item.get("role") not in {"user", "assistant"} for item in items):
        return False, "unsupported_message_role"
    if any(not isinstance(item.get("content"), str) for item in items):
        return False, "message_content_must_be_text"
    if not any(item["role"] == "user" for item in items):
        return False, "missing_user_message"
    if not any(item["role"] == "assistant" for item in items):
        return False, "missing_assistant_message"
    return True, None


def _image_placeholders(messages: Iterable[dict[str, Any]]) -> int:
    return sum(item.get("content", "").count("<image>") for item in messages)


def validate_supervision_unit(unit: SupervisionUnit) -> ValidationDecision:
    """Validate one internal target without making it a training ExportRow."""

    valid_shape, shape_reason = _message_shape_valid(unit.context_messages)
    if not valid_shape:
        return ValidationDecision(
            status="review_required",
            supervision_type=unit.supervision_type,
            reasons=(shape_reason or "invalid_context",),
            evidence_sources=unit.evidence_sources,
        )

    if unit.supervision_type == "solver_positive":
        reasons: list[str] = []
        if unit.controlled_ancestry:
            reasons.append("controlled_ancestry")
        if unit.repaired_error:
            reasons.append("repaired_error")
        if unit.regenerated:
            reasons.append("regenerated_target")
        if reasons:
            return ValidationDecision(
                status="rejected",
                supervision_type="solver_positive",
                reasons=tuple(reasons),
                evidence_sources=unit.evidence_sources,
            )
        return ValidationDecision(
            status="accepted",
            supervision_type="solver_positive",
            evidence_sources=unit.evidence_sources,
            solver_positive_eligible=True,
        )

    if unit.supervision_type not in {"verifier", "repair", "regeneration"}:
        return ValidationDecision(
            status="rejected",
            supervision_type=unit.supervision_type,
            reasons=("unsupported_supervision_type",),
        )

    if not LOCAL_EVIDENCE_SOURCES.intersection(unit.evidence_sources):
        return ValidationDecision(
            status="review_required",
            supervision_type=unit.supervision_type,
            reasons=("missing_independent_local_evidence",),
            evidence_sources=unit.evidence_sources,
        )
    # These units are valid audit artifacts, but the role policy deliberately
    # keeps them out of the final Solver JSONL.
    return ValidationDecision(
        status="accepted",
        supervision_type=unit.supervision_type,
        evidence_sources=unit.evidence_sources,
        solver_positive_eligible=False,
        exportable=False,
    )


def validate_target_repair_depth(record: TrajectoryRecord) -> None:
    if record.target_repair_depth != record.target_repair_depth_frozen:
        raise ValueError(
            "target_repair_depth was changed after Task Analysis: "
            f"{record.target_repair_depth_frozen} -> {record.target_repair_depth}"
        )
    if not 0 <= record.target_repair_depth <= 2:
        raise ValueError("target_repair_depth must be in [0, 2]")
    if record.canonical_actual_repair_count < 0 or record.branch_actual_repair_count < 0:
        raise ValueError("repair counts must be non-negative")


def repair_depth_stats(record: TrajectoryRecord) -> RepairDepthStats:
    validate_target_repair_depth(record)
    return RepairDepthStats(
        target_repair_depth=record.target_repair_depth,
        canonical_actual_repair_count=record.canonical_actual_repair_count,
        branch_actual_repair_count=record.branch_actual_repair_count,
    )


def ancestry_is_contaminated(record: TrajectoryRecord) -> bool:
    """Inspect explicit lineage facts, not merely strings in final messages."""

    return bool(
        record.has_controlled_ancestry
        or CONTROLLED_EVENTS.intersection(record.lineage_events)
        or record.audit_context_received
        or record.controlled_intervention
    )


def validate_trajectory_for_export(
    record: TrajectoryRecord,
    *,
    source_guard: SourceGuardDecision | None = None,
    max_reasoning_steps: int = 8,
) -> tuple[ValidationDecision, ExportCandidate | None]:
    """Return an export candidate only for a clean, natural Solver lineage."""

    try:
        validate_target_repair_depth(record)
    except ValueError as exc:
        return (
            ValidationDecision(
                status="review_required",
                supervision_type="solver_positive",
                reasons=(str(exc),),
            ),
            None,
        )

    reasons: list[str] = []
    shape_ok, shape_reason = _message_shape_valid(record.messages)
    if not shape_ok:
        reasons.append(shape_reason or "invalid_message_shape")
    if _image_placeholders(record.messages) != len(record.images):
        reasons.append("image_placeholder_count_mismatch")
    if len(record.images) != len(record.image_content_hashes):
        reasons.append("image_hash_count_mismatch")
    try:
        final_assistant = next(
            item["content"]
            for item in reversed(record.messages)
            if item.get("role") == "assistant"
        )
        validate_solver_final(final_assistant, max_reasoning_steps=max_reasoning_steps)
    except (StopIteration, ProtocolError) as exc:
        reasons.append(f"invalid_solver_final:{exc}")
    if not record.final_answer_valid:
        reasons.append("final_answer_not_independently_validated")
    if not record.clean_natural_lineage:
        reasons.append("not_clean_natural_lineage")
    if ancestry_is_contaminated(record):
        reasons.append("controlled_ancestry")
    if any(not valid for valid in record.step_validated):
        reasons.append("unvalidated_solver_step")
    if record.repaired_error_steps:
        reasons.append("repaired_error_present")
    if record.regenerated_steps:
        reasons.append("regeneration_present")
    if source_guard is None:
        reasons.append("missing_source_guard_decision")
    elif not source_guard.accepted:
        reasons.extend(source_guard.reasons or ("source_guard_rejected",))

    if reasons:
        return (
            ValidationDecision(
                status="rejected" if "controlled_ancestry" in reasons else "review_required",
                supervision_type="solver_positive",
                reasons=tuple(dict.fromkeys(reasons)),
                solver_positive_eligible=False,
                exportable=False,
            ),
            None,
        )

    decision = ValidationDecision(
        status="accepted",
        supervision_type="solver_positive",
        evidence_sources=("deterministic_rule",),
        solver_positive_eligible=True,
        exportable=True,
    )
    return (
        decision,
        ExportCandidate(
            task_id=record.task_id,
            trajectory_id=record.trajectory_id,
            source_record_id=record.source_record_id,
            stage=record.stage,
            messages=tuple(record.messages),
            images=tuple(record.images),
            image_content_hashes=tuple(record.image_content_hashes),
            validation=decision,
        ),
    )
