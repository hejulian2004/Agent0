from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.local_sft_builder.canonical import sha256_file
from tools.local_sft_builder.source_adapter import (
    DuplicateSourceIdentityError,
    ForbiddenIndexError,
    ForbiddenSourceIndex,
    InvalidImageRootError,
    RealSourceAdapter,
    SourceInputError,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row(
    *,
    task_id: str = "task-1",
    source_record_id: str | None = None,
    original_id: str = "original-1",
    source_dataset: str = "fixture",
    source_revision: str = "fixture-v1",
    split: str = "train",
    usage_partition: str = "sft_stage1",
    question: str = "What is 2 + 2?",
    ground_truth: str = "4",
    images: list[str] | None = None,
    image_refs: list[dict] | None = None,
    **extra,
) -> dict:
    if source_record_id is None:
        source_record_id = task_id
    row = {
        "task_id": task_id,
        "source_record_id": source_record_id,
        "original_id": original_id,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "split": split,
        "usage_partition": usage_partition,
        "question": question,
        "ground_truth": ground_truth,
        "images": [] if images is None else images,
        "image_refs": [] if image_refs is None else image_refs,
    }
    row.update(extra)
    return row


def _visual_row(
    image_root: Path,
    *,
    task_id: str = "visual-task",
    source_record_id: str = "visual-record",
    original_id: str = "visual-original",
    image_name: str = "images/item.png",
    content: bytes = b"phase2a image",
    question: str = "<image>\nDescribe the image.",
) -> dict:
    image_path = image_root / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(content)
    return _row(
        task_id=task_id,
        source_record_id=source_record_id,
        original_id=original_id,
        question=question,
        images=[image_name],
        image_refs=[
            {
                "asset_id": f"fixture/{task_id}/image-0",
                "content_sha256": sha256_file(image_path),
            }
        ],
    )


def _adapter(
    source_path: Path,
    *,
    image_root: Path | None = None,
    image_root_config: str | None = None,
    forbidden_index: ForbiddenSourceIndex | None = None,
) -> RealSourceAdapter:
    return RealSourceAdapter(
        source_path,
        stage="stage1",
        image_root=image_root,
        image_root_config=image_root_config,
        forbidden_index=forbidden_index,
    )


def test_source_task_preserves_distinct_record_and_original_ids(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [_row(source_record_id="builder-record", original_id="upstream-sample")],
    )

    result = _adapter(source_path).preflight(seed=0, max_tasks=1)

    task = result.selected_tasks[0]
    assert task.source_record_id == "builder-record"
    assert task.original_id == "upstream-sample"
    assert task.as_source_record()["source_record_id"] == "builder-record"
    assert task.as_source_record()["original_id"] == "upstream-sample"


def test_empty_ground_truth_is_allowed_but_null_is_review_required(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            _row(task_id="empty", original_id="empty", ground_truth=""),
            _row(task_id="null", original_id="null", ground_truth=None),
        ],
    )

    result = _adapter(source_path).preflight(seed=0, max_tasks=10)

    assert result.accepted_before_sampling == 1
    assert result.counts["review_required"] == 1
    assert result.selected_tasks[0].ground_truth == ""


def test_visual_source_verifies_physical_hash_and_keeps_namespaces_separate(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "image-root"
    source_path = tmp_path / "source.jsonl"
    row = _visual_row(image_root)
    _write_jsonl(source_path, [row])

    result = _adapter(
        source_path,
        image_root=image_root,
        image_root_config="fixture-images-v1",
    ).preflight(seed=0, max_tasks=1)

    task = result.selected_tasks[0]
    assert task.images == (str((image_root / "images/item.png").resolve()),)
    assert task.image_refs[0]["asset_id"] == "fixture/visual-task/image-0"
    assert task.image_content_hashes == (row["image_refs"][0]["content_sha256"],)
    assert result.entries[0].image_content_hashes == task.image_content_hashes


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_revision", None),
        ("source_revision", "   "),
        ("source_revision", "unknown"),
        ("split", None),
        ("usage_partition", "   "),
        ("original_id", ""),
        ("source_record_id", ""),
        ("ground_truth", None),
    ),
)
def test_missing_or_blank_row_metadata_is_review_required(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [_row(**{field: value})])

    result = _adapter(source_path).preflight(seed=0, max_tasks=1)

    assert result.counts["review_required"] == 1
    assert result.selected_tasks == ()


@pytest.mark.parametrize(
    "bad_ref",
    (
        {"asset_id": "fixture/image-0"},
        {"asset_id": "fixture/image-0", "content_sha256": None},
        {"asset_id": "fixture/image-0", "content_sha256": "A" * 64},
        {"asset_id": "fixture/image-0", "content_sha256": "a" * 63},
        {"asset_id": "fixture/image-0", "content_sha256": "a" * 65},
    ),
)
def test_invalid_visual_metadata_is_review_required(
    tmp_path: Path,
    bad_ref: dict,
) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [_row(question="<image>\nQ", images=["image.png"], image_refs=[bad_ref])],
    )

    result = _adapter(source_path, image_root=tmp_path, image_root_config="fixture").preflight(
        seed=0,
        max_tasks=1,
    )

    assert result.counts["review_required"] == 1
    assert result.selected_tasks == ()


def test_visual_source_without_image_root_is_review_required(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            _row(
                question="<image>\nQ",
                images=["image.png"],
                image_refs=[
                    {"asset_id": "fixture/image", "content_sha256": "a" * 64}
                ],
            )
        ],
    )

    result = _adapter(source_path).preflight(seed=0, max_tasks=1)

    assert result.counts["review_required"] == 1
    assert result.selected_tasks == ()


def test_image_count_and_placeholder_mismatches_are_review_required(tmp_path: Path) -> None:
    image_root = tmp_path / "image-root"
    row = _visual_row(image_root)
    row["images"] = ["images/item.png", "images/item.png"]
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [row])

    result = _adapter(
        source_path,
        image_root=image_root,
        image_root_config="fixture",
    ).preflight(seed=0, max_tasks=1)
    assert result.counts["review_required"] == 1

    row = _visual_row(image_root, task_id="placeholder", original_id="placeholder")
    row["question"] = "Q without an image placeholder"
    _write_jsonl(source_path, [row])
    result = _adapter(
        source_path,
        image_root=image_root,
        image_root_config="fixture",
    ).preflight(seed=0, max_tasks=1)
    assert result.counts["review_required"] == 1


def test_image_absolute_and_parent_paths_are_review_required(tmp_path: Path) -> None:
    image_root = tmp_path / "image-root"
    image_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    valid_hash = sha256_file(outside)
    source_path = tmp_path / "source.jsonl"
    rows = [
        _row(
            task_id="absolute",
            original_id="absolute",
            question="<image>\nQ",
            images=[str(outside)],
            image_refs=[{"asset_id": "fixture/absolute", "content_sha256": valid_hash}],
        ),
        _row(
            task_id="parent",
            original_id="parent",
            question="<image>\nQ",
            images=["../outside.png"],
            image_refs=[{"asset_id": "fixture/parent", "content_sha256": valid_hash}],
        ),
    ]
    _write_jsonl(source_path, rows)

    result = _adapter(
        source_path,
        image_root=image_root,
        image_root_config="fixture",
    ).preflight(seed=0, max_tasks=10)
    assert result.counts["review_required"] == 2


def test_symlink_escape_is_review_required_when_supported(tmp_path: Path) -> None:
    image_root = tmp_path / "image-root"
    image_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    link = image_root / "link.png"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            _row(
                question="<image>\nQ",
                images=["link.png"],
                image_refs=[
                    {"asset_id": "fixture/link", "content_sha256": sha256_file(outside)}
                ],
            )
        ],
    )
    result = _adapter(
        source_path,
        image_root=image_root,
        image_root_config="fixture",
    ).preflight(seed=0, max_tasks=1)
    assert result.counts["review_required"] == 1


def test_image_content_hash_mismatch_is_review_required(tmp_path: Path) -> None:
    image_root = tmp_path / "image-root"
    image_root.mkdir()
    image_path = image_root / "image.png"
    image_path.write_bytes(b"actual")
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            _row(
                question="<image>\nQ",
                images=["image.png"],
                image_refs=[
                    {"asset_id": "fixture/image", "content_sha256": "a" * 64}
                ],
            )
        ],
    )
    result = _adapter(
        source_path,
        image_root=image_root,
        image_root_config="fixture",
    ).preflight(seed=0, max_tasks=1)
    assert result.counts["review_required"] == 1


@pytest.mark.parametrize("source_content_hash", (None, "", "A" * 64, "a" * 63))
def test_invalid_explicit_source_content_hash_is_review_required(
    tmp_path: Path,
    source_content_hash: object,
) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [_row(source_content_hash=source_content_hash)],
    )

    result = _adapter(source_path).preflight(seed=0, max_tasks=1)

    assert result.counts["review_required"] == 1


def test_explicit_and_derived_source_content_hash_origins_are_recorded(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            _row(task_id="derived", original_id="derived"),
            _row(
                task_id="declared",
                original_id="declared",
                source_content_hash="b" * 64,
            ),
        ],
    )

    result = _adapter(source_path).preflight(seed=0, max_tasks=10)

    origins = {
        entry.task_id: entry.source_content_hash_origin
        for entry in result.entries
    }
    assert origins == {"derived": "derived", "declared": "declared"}


def test_duplicate_stable_ids_abort_across_all_source_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [_row(task_id="same-task", original_id="one")])
    _write_jsonl(second, [_row(task_id="same-task", original_id="two")])

    with pytest.raises(DuplicateSourceIdentityError):
        _adapter((first, second)).preflight(seed=0, max_tasks=10)


def test_duplicate_task_id_aborts_even_when_later_row_has_bad_image(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            _row(task_id="same-task", original_id="first"),
            _row(
                task_id="same-task",
                original_id="second",
                question="<image>\nQ",
                images=["missing.png"],
                image_refs=[
                    {"asset_id": "fixture/missing", "content_sha256": "a" * 64}
                ],
            ),
        ],
    )

    with pytest.raises(DuplicateSourceIdentityError):
        _adapter(source_path).preflight(seed=0, max_tasks=10)


@pytest.mark.parametrize(
    "rows",
    (
        [
            _row(task_id="one", original_id="same"),
            _row(task_id="two", original_id="same"),
        ],
        [
            _row(task_id="one", source_record_id="same"),
            _row(task_id="two", source_record_id="same", original_id="two"),
        ],
    ),
)
def test_duplicate_dataset_identity_aborts(
    tmp_path: Path,
    rows: list[dict],
) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, rows)
    with pytest.raises(DuplicateSourceIdentityError):
        _adapter(source_path).preflight(seed=0, max_tasks=10)


def test_same_source_record_id_across_datasets_is_allowed(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            _row(
                task_id="one",
                source_record_id="shared-record",
                source_dataset="dataset-a",
                original_id="one",
            ),
            _row(
                task_id="two",
                source_record_id="shared-record",
                source_dataset="dataset-b",
                original_id="two",
            ),
        ],
    )

    result = _adapter(source_path).preflight(seed=0, max_tasks=10)

    assert result.accepted_before_sampling == 2


def test_forbidden_index_accepts_non_train_rows_without_train_guard(tmp_path: Path) -> None:
    forbidden_path = tmp_path / "forbidden.jsonl"
    _write_jsonl(
        forbidden_path,
        [
            _row(task_id="test-task", original_id="test", split="test", usage_partition="eval"),
            _row(
                task_id="validation-task",
                original_id="validation",
                split="validation",
                usage_partition="reserve",
            ),
        ],
    )

    index = ForbiddenSourceIndex.from_paths((forbidden_path,))

    assert index.record_count == 2


def test_forbidden_identity_overlap_is_rejected(tmp_path: Path) -> None:
    forbidden_path = tmp_path / "forbidden.jsonl"
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        forbidden_path,
        [_row(task_id="forbidden-task", original_id="same-original", split="test")],
    )
    _write_jsonl(
        source_path,
        [_row(task_id="train-task", original_id="same-original")],
    )
    index = ForbiddenSourceIndex.from_paths((forbidden_path,))

    result = _adapter(source_path, forbidden_index=index).preflight(seed=0, max_tasks=1)

    assert result.entries[0].status == "rejected"
    assert result.entries[0].reasons == ("forbidden_source_identity_overlap",)
    assert result.selected_tasks == ()


def test_forbidden_identity_corruption_aborts_preflight(tmp_path: Path) -> None:
    forbidden_path = tmp_path / "forbidden.jsonl"
    _write_jsonl(forbidden_path, [_row(source_revision=None, split="test")])

    with pytest.raises(ForbiddenIndexError):
        ForbiddenSourceIndex.from_paths((forbidden_path,))


def test_all_rows_are_decided_before_hash_ranking_sampling(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(
        source_path,
        [
            _row(task_id="accepted-a", original_id="accepted-a"),
            _row(task_id="bad-row", original_id="bad-row", source_revision=" "),
            _row(task_id="accepted-b", original_id="accepted-b"),
        ],
    )

    result = _adapter(source_path).preflight(seed=0, max_tasks=1)

    assert result.total_source_rows == 3
    assert result.accepted_before_sampling == 2
    assert result.selected_after_sampling == 1
    assert result.counts["review_required"] == 1
    assert {entry.task_id for entry in result.entries} == {
        "accepted-a",
        "bad-row",
        "accepted-b",
    }


def test_hash_ranking_is_explicit_and_reproducible(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    rows = [_row(task_id=f"task-{index}", original_id=f"original-{index}") for index in range(8)]
    _write_jsonl(source_path, rows)

    first = _adapter(source_path).preflight(seed=17, max_tasks=3)
    second = _adapter(source_path).preflight(seed=17, max_tasks=3)

    expected = sorted(
        (
            hashlib.sha256(f"17\0{row['task_id']}".encode("utf-8")).hexdigest(),
            row["task_id"],
        )
        for row in rows
    )[:3]
    assert [task.task_id for task in first.selected_tasks] == [item[1] for item in expected]
    assert [task.task_id for task in second.selected_tasks] == [task.task_id for task in first.selected_tasks]
    assert [entry.to_dict() for entry in first.entries] == [entry.to_dict() for entry in second.entries]


def test_malformed_source_jsonl_aborts(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"task_id": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(SourceInputError):
        _adapter(source_path).preflight(seed=0, max_tasks=1)


def test_invalid_image_root_configuration_aborts(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageRootError):
        RealSourceAdapter(
            tmp_path / "source.jsonl",
            stage="stage1",
            image_root=tmp_path / "does-not-exist",
            image_root_config="fixture",
        )
