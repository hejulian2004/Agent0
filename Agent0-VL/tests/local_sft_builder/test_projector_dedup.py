from __future__ import annotations

import pytest

from tools.local_sft_builder.dedup import DedupError, exact_dedup_key, exact_deduplicate
from tools.local_sft_builder.projector import Projector
from tools.local_sft_builder.schema import ExportCandidate, ExportRow, ValidationDecision


def _candidate(*, status="accepted", supervision_type="solver_positive"):
    decision = ValidationDecision(
        status=status,
        supervision_type=supervision_type,
        solver_positive_eligible=status == "accepted" and supervision_type == "solver_positive",
        exportable=status == "accepted" and supervision_type == "solver_positive",
    )
    return ExportCandidate(
        task_id="t1",
        trajectory_id="tr1",
        source_record_id="o1",
        stage="sft_stage1",
        messages=(
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "final"},
        ),
        images=(),
        image_content_hashes=(),
        validation=decision,
    )


def test_projector_only_consumes_validator_accepted_candidate() -> None:
    row = Projector().project(_candidate())
    assert row is not None
    assert set(row.to_dict()) == {"messages", "images"}
    assert len([item for item in row.messages if item["role"] == "assistant"]) == 2

    assert Projector().project(_candidate(status="audit_only")) is None
    assert Projector().project(_candidate(supervision_type="repair")) is None


def test_exact_dedup_is_path_independent_and_stage_global() -> None:
    messages = (
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    )
    stage2 = ExportRow(
        messages=messages,
        images=("/mnt/d/item.png",),
        image_content_hashes=("b" * 64,),
        source_record_id="z-source",
        trajectory_id="z-traj",
        stage="sft_stage2",
    )
    stage1 = ExportRow(
        messages=messages,
        images=(r"D:\dataset\item.png",),
        image_content_hashes=("b" * 64,),
        source_record_id="a-source",
        trajectory_id="a-traj",
        stage="sft_stage1",
    )
    assert exact_dedup_key(stage1) == exact_dedup_key(stage2)
    assert exact_deduplicate([stage2, stage1]) == [stage1]
    assert exact_deduplicate([stage1, stage2]) == [stage1]


def test_exact_dedup_requires_image_content_hashes() -> None:
    row = ExportRow(
        messages=({"role": "user", "content": "Q"},),
        images=("image.png",),
    )
    with pytest.raises(DedupError):
        exact_dedup_key(row)
