"""Validator-gated projection to the upstream Swift row shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import ExportCandidate, ExportRow


class ProjectionError(ValueError):
    """Raised for a structurally invalid export candidate."""


def _validate_export_shape(messages: tuple[dict, ...], images: tuple[str, ...]) -> None:
    if not messages:
        raise ProjectionError("export candidate has no messages")
    if any(item.get("role") not in {"user", "assistant"} for item in messages):
        raise ProjectionError("export rows may contain only user and assistant roles")
    if any(not isinstance(item.get("content"), str) for item in messages):
        raise ProjectionError("export message content must be text")
    if not any(item["role"] == "user" for item in messages):
        raise ProjectionError("export candidate has no user message")
    if not any(item["role"] == "assistant" for item in messages):
        raise ProjectionError("export candidate has no assistant message")
    placeholders = sum(item["content"].count("<image>") for item in messages)
    if placeholders != len(images):
        raise ProjectionError(
            f"image placeholder count {placeholders} != image count {len(images)}"
        )


class Projector:
    """A deliberately policy-free projector.

    Quality and eligibility are decided by Validator. This class only checks
    the validator decision and the mechanical final-row shape.
    """

    def project(self, candidate: ExportCandidate) -> ExportRow | None:
        decision = candidate.validation
        if not (
            decision.status == "accepted"
            and decision.supervision_type == "solver_positive"
            and decision.solver_positive_eligible
            and decision.exportable
        ):
            return None
        _validate_export_shape(candidate.messages, candidate.images)
        return ExportRow(
            messages=tuple(candidate.messages),
            images=tuple(candidate.images),
            image_content_hashes=tuple(candidate.image_content_hashes),
            source_record_id=candidate.source_record_id,
            trajectory_id=candidate.trajectory_id,
            stage=candidate.stage,
        )

    def project_many(self, candidates: Iterable[ExportCandidate]) -> list[ExportRow]:
        rows: list[ExportRow] = []
        for candidate in candidates:
            row = self.project(candidate)
            if row is not None:
                rows.append(row)
        return rows


def write_export_jsonl(rows: Iterable[ExportRow], path: str | Path) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count
