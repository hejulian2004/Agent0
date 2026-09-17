"""Policy-compliant validation → projection → dedup orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .dedup import exact_deduplicate
from .projector import Projector, write_export_jsonl
from .schema import ExportRow, TrajectoryRecord
from .source_guard import SourceGuardDecision
from .validator import validate_trajectory_for_export


@dataclass(frozen=True)
class PipelineResult:
    rows_before_dedup: tuple[ExportRow, ...]
    rows_after_dedup: tuple[ExportRow, ...]
    validation_decisions: tuple[object, ...]


def validate_project_deduplicate(
    trajectories: Iterable[TrajectoryRecord],
    source_decisions: Mapping[str, SourceGuardDecision],
) -> PipelineResult:
    """Apply Validator policy before Projector and exact dedup.

    The Projector receives only candidates created by Validator; it never
    receives raw trajectories and therefore cannot reimplement quality policy.
    """

    candidates = []
    decisions = []
    for trajectory in trajectories:
        decision, candidate = validate_trajectory_for_export(
            trajectory,
            source_guard=source_decisions.get(trajectory.task_id),
        )
        decisions.append(decision)
        if candidate is not None:
            candidates.append(candidate)
    rows = Projector().project_many(candidates)
    deduped = exact_deduplicate(rows)
    return PipelineResult(tuple(rows), tuple(deduped), tuple(decisions))


def export_pipeline_result(result: PipelineResult, path: str | Path) -> int:
    return write_export_jsonl(result.rows_after_dedup, path)
