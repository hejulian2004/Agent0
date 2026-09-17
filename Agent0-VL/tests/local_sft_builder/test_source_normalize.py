"""Tests for the formal-partition source normalizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.local_sft_builder.canonical import sha256_file
from tools.local_sft_builder.source_adapter import (
    ForbiddenSourceIndex,
    RealSourceAdapter,
)
from tools.local_sft_builder.source_guard import SourceLeakageGuard
from tools.local_sft_builder.source_normalize import (
    ANSWER_CHECK_METHOD,
    EXPECTED_RETOOL_REVISION,
    GROUND_TRUTH_ORIGIN,
    REVISION_ORIGIN_DECLARED,
    REVISION_ORIGIN_PINNED,
    RawSourceError,
    RowReviewRequired,
    UnsupportedDatasetError,
    build_forbidden_rows,
    extract_mulberry,
    extract_mulberry_final_answer,
    extract_retool,
    iter_json_array,
    normalize_stage,
)


REVISION = "486ed06f81be192a14ca7f71ea54d81db1ab084e"
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"local-sft-builder-fixture"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _write_json_array(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def _mulberry_record(
    *,
    image: str = "cauldron/clevr/images/clevr_00011395.png",
    answer: str = "3",
    question: str = "<image>\nQuestion: how many cubes are there?",
) -> dict:
    return {
        "images": image,
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": (
                    "### Image Description:\nA small scene.\n\n"
                    f"### The final answer is: {answer}"
                ),
            },
        ],
    }


def _retool_record(
    *,
    question: str = "What is 17 * 3?",
    answer: str = "51",
) -> dict:
    """A ReTool row shaped like the real ``train_2000.parquet``.

    The real rows are multi-turn: a ``system`` turn, a ``user`` turn carrying the
    ``**user question:**`` marker plus the dataset's own trailing format
    instruction, ``assistant`` turns that request tool calls, ``tool`` turns with
    the interpreter output, and a final ``assistant`` turn whose reference answer
    is wrapped as ``<answer>\\boxed{...}</answer>``.  Everything before that
    wrapper is the reference trajectory's reasoning, so the raw assistant text
    must never be used as the reference label.

    The earlier fixture used a bare two-message record with a bare numeric
    answer, which is why the reference-answer defect below stayed invisible.
    """

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can solve math problems "
                    "with interaction Code Interpreter by Python code."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Solve the following problem step by step. You now have the "
                    "ability to selectively write executable Python code to "
                    "enhance your reasoning process.\n\n"
                    f"**user question:**\n{question}\n\n"
                    "Remember to place the final answer in the last part using "
                    "the format: \n<answer>\noxed{'The final answer goes "
                    "here.'}\n</answer>"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Okay, so I need to work this out step by step. Let me compute "
                    "it using code to ensure accuracy."
                ),
                "tool_calls": [{"id": "call_0", "type": "function"}],
            },
            {"role": "tool", "content": "17", "tool_call_id": "call_0"},
            {
                "role": "assistant",
                "content": (
                    "The verification confirms the intermediate result, so the "
                    "reasoning holds.\n\n"
                    f"<answer>\n\\boxed{{{answer}}}\n</answer>"
                ),
            },
        ]
    }


def _formal_row(
    *,
    dataset: str,
    original_id: str,
    row_index: int,
    image_ref: str | None = None,
    license_status: str = "verified",
    eligible: bool = True,
    split: str = "train",
    usage_partition: str = "sft_stage1",
    revision: str = REVISION,
) -> dict:
    row = {
        "dataset": dataset,
        "original_id": original_id,
        "source_record_id": f"{dataset}:{original_id}",
        "resolved_revision": revision,
        "split": split,
        "usage_partition": usage_partition,
        "row_index": row_index,
        "license_status": license_status,
        "formal_training_eligible": eligible,
    }
    if image_ref is not None:
        row["image_ref"] = image_ref
    return row


@pytest.fixture()
def staged(tmp_path: Path) -> dict:
    """A minimal on-disk staging tree plus its normalized inputs."""

    raw_root = tmp_path / "staging"
    image_root = raw_root / "mulberry-proxy" / "mulberry_images"
    (image_root / "cauldron" / "clevr" / "images").mkdir(parents=True)
    image_bytes = _PNG_BYTES
    (image_root / "cauldron" / "clevr" / "images" / "clevr_00011395.png").write_bytes(
        image_bytes
    )

    _write_json_array(
        raw_root / "mulberry-proxy" / "mulberry_sft.json",
        [
            _mulberry_record(answer="1"),
            _mulberry_record(answer="3"),
            _mulberry_record(answer="7"),
        ],
    )
    retool_path = _write_json_array(
        raw_root / "retool.json",
        [_retool_record(question="What is 17 * 3?", answer="51")],
    )

    formal_path = _write_jsonl(
        tmp_path / "formal_sft_stage1.jsonl",
        [
            _formal_row(
                dataset="mulberry",
                original_id="mulberry-260494",
                row_index=1,
                image_ref="cauldron/clevr/images/clevr_00011395.png",
            ),
            _formal_row(
                dataset="retool",
                original_id="retool-2",
                row_index=0,
            ),
        ],
    )
    return {
        "tmp_path": tmp_path,
        "raw_root": raw_root,
        "image_root": image_root,
        "image_bytes": image_bytes,
        "formal_path": formal_path,
        "retool_path": retool_path,
    }


# --------------------------------------------------------------------------
# Streaming reader
# --------------------------------------------------------------------------


def test_iter_json_array_streams_with_tiny_buffer(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    payload = [
        {"a": 1, "nested": {"b": [1, 2, 3]}},
        {"a": 2, "text": "contains ] and [ braces"},
        {"a": 3},
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    assert list(iter_json_array(path, read_size=7)) == payload


def test_iter_json_array_rejects_non_array(tmp_path: Path) -> None:
    path = tmp_path / "object.json"
    path.write_text('{"not": "an array"}', encoding="utf-8")

    with pytest.raises(RawSourceError):
        list(iter_json_array(path))


def test_iter_json_array_rejects_malformed_element(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('[{"a": 1}, {"b": ]}', encoding="utf-8")

    with pytest.raises(RawSourceError):
        list(iter_json_array(path))


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_extract_mulberry_uses_last_final_answer_marker() -> None:
    record = {
        "images": "DVQA/images/bar_train_00134926.png",
        "messages": [
            {"role": "user", "content": "<image>\nQuestion: value?"},
            {
                "role": "assistant",
                "content": (
                    "The final answer is: placeholder\n"
                    "more reasoning\n"
                    "### The final answer is: 12"
                ),
            },
        ],
    }

    extracted = extract_mulberry(record)

    assert extracted.ground_truth == "12"
    assert extracted.images == ("DVQA/images/bar_train_00134926.png",)
    assert extracted.question.count("<image>") == 1


def test_extract_mulberry_final_answer_strips_quotes() -> None:
    assert extract_mulberry_final_answer("### The final answer is: 'yes'") == "yes"
    assert extract_mulberry_final_answer("no marker here") is None


def test_extract_mulberry_without_answer_marker_raises() -> None:
    record = {
        "images": "cauldron/clevr/images/clevr_00000001.png",
        "messages": [
            {"role": "user", "content": "<image>\nQuestion: value?"},
            {"role": "assistant", "content": "I am not sure."},
        ],
    }

    with pytest.raises(RowReviewRequired) as excinfo:
        extract_mulberry(record)

    assert excinfo.value.code == "mulberry_reference_answer_not_parseable"


def test_extract_retool_is_text_only() -> None:
    extracted = extract_retool(_retool_record(question="What is 17 * 3?", answer="51"))

    # ``extract_retool_question`` keeps everything after the marker verbatim,
    # including the source dataset's own trailing format instruction.
    assert extracted.question.startswith("What is 17 * 3?")
    assert extracted.ground_truth == "51"
    assert extracted.images == ()
    assert extracted.question.count("<image>") == 0


def test_extract_retool_ground_truth_is_not_the_raw_reasoning_text() -> None:
    """Regression: the whole assistant turn must never become the label.

    The real ReTool rows end with ``<answer>\\boxed{...}</answer>`` after a long
    reasoning prefix.  Taking the raw text made every one of the 2000 rows fail
    the reference-answer check, and therefore the export gate, with a 0% yield.
    """

    record = _retool_record(question="What is 17 * 3?", answer="51")
    assistant_text = record["messages"][-1]["content"]

    assert "\\boxed{51}" in assistant_text
    assert len(assistant_text) > 100  # the reference trajectory's reasoning
    assert extract_retool(record).ground_truth == "51"


def test_extract_retool_reference_falls_back_to_the_answer_block() -> None:
    record = _retool_record(question="What is 17 * 3?", answer="51")
    record["messages"][-1]["content"] = (
        "The reasoning is complete.\n\n<answer>\n51\n</answer>"
    )

    assert extract_retool(record).ground_truth == "51"


def test_extract_retool_without_any_reference_answer_raises() -> None:
    record = _retool_record(question="What is 17 * 3?", answer="51")
    record["messages"][-1]["content"] = "The reasoning is complete."

    with pytest.raises(RowReviewRequired) as excinfo:
        extract_retool(record)

    assert excinfo.value.code == "retool_reference_answer_not_parseable"


def test_extract_retool_without_marker_raises() -> None:
    with pytest.raises(RowReviewRequired) as excinfo:
        extract_retool({"messages": [{"role": "user", "content": "no marker"}]})

    assert excinfo.value.code == "retool_question_marker_not_found"


# --------------------------------------------------------------------------
# End-to-end normalization
# --------------------------------------------------------------------------


def test_normalized_rows_pass_real_source_adapter(staged: dict) -> None:
    rows, stats = normalize_stage(
        formal_source=staged["formal_path"],
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10, "retool": 10},
        raw_overrides={"retool": staged["retool_path"]},
    )

    assert stats.review_required == {}
    assert [row.source_dataset for row in rows] == ["mulberry", "retool"]

    mulberry_row, retool_row = rows
    assert mulberry_row.task_id == "mulberry-stage1-000001"
    assert mulberry_row.ground_truth == "3"
    assert mulberry_row.image_refs == (
        {
            "asset_id": "mulberry/mulberry-260494/image-0",
            "content_sha256": sha256_file(
                staged["image_root"]
                / "cauldron/clevr/images/clevr_00011395.png"
            ),
        },
    )
    # ReTool must stay text-only so the single image_root invariant holds.
    assert retool_row.images == ()
    assert retool_row.image_refs == ()
    assert retool_row.question.count("<image>") == 0

    source_path = staged["tmp_path"] / "source_stage1.jsonl"
    _write_jsonl(source_path, [row.to_dict() for row in rows])

    adapter = RealSourceAdapter(
        source_path,
        stage="stage1",
        image_root=staged["image_root"],
        image_root_config="mulberry_images_v1",
        source_guard=SourceLeakageGuard(),
    )
    result = adapter.preflight(seed=0, max_tasks=10)

    assert result.counts == {"accepted": 2, "rejected": 0, "review_required": 0}
    assert result.selected_after_sampling == 2
    selected = {task.source_dataset: task for task in result.selected_tasks}
    assert selected["mulberry"].image_content_hashes == (
        sha256_file(staged["image_root"] / "cauldron/clevr/images/clevr_00011395.png"),
    )
    assert selected["retool"].images == ()

    # The runtime needs the resolved path, but the exported SFT row must stay
    # image-root relative so the dataset can be moved to another machine.
    mulberry_task = selected["mulberry"]
    assert Path(mulberry_task.images[0]) == (
        staged["image_root"] / "cauldron/clevr/images/clevr_00011395.png"
    )
    assert mulberry_task.image_relatives == (
        "cauldron/clevr/images/clevr_00011395.png",
    )
    assert mulberry_task.export_images == (
        "cauldron/clevr/images/clevr_00011395.png",
    )
    assert selected["retool"].export_images == ()


def test_normalization_is_deterministic(staged: dict) -> None:
    kwargs = {
        "formal_source": staged["formal_path"],
        "stage": "stage1",
        "usage_partition": "sft_stage1",
        "raw_root": staged["raw_root"],
        "image_root": staged["image_root"],
        "per_dataset_limit": {"mulberry": 10, "retool": 10},
        "raw_overrides": {"retool": staged["retool_path"]},
    }

    first, _ = normalize_stage(**kwargs)
    second, _ = normalize_stage(**kwargs)

    assert [row.to_dict() for row in first] == [row.to_dict() for row in second]


def test_provenance_is_recorded_on_every_row(staged: dict) -> None:
    rows, _ = normalize_stage(
        formal_source=staged["formal_path"],
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10, "retool": 10},
        raw_overrides={"retool": staged["retool_path"]},
    )

    for row in rows:
        payload = row.to_dict()
        assert payload["ground_truth_origin"] == GROUND_TRUTH_ORIGIN
        assert payload["answer_check_method"] == ANSWER_CHECK_METHOD
        assert payload["image_count"] == len(payload["images"])


def test_unverified_license_is_review_required(staged: dict) -> None:
    staged["formal_path"] = _write_jsonl(
        staged["tmp_path"] / "unverified.jsonl",
        [
            _formal_row(
                dataset="mulberry",
                original_id="mulberry-1",
                row_index=1,
                license_status="manual_review_required",
                eligible=False,
            )
        ],
    )

    rows, stats = normalize_stage(
        formal_source=staged["formal_path"],
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10},
    )

    assert rows == []
    assert stats.review_required == {"formal_license_not_verified": 1}


def test_non_train_split_is_review_required(staged: dict) -> None:
    staged["formal_path"] = _write_jsonl(
        staged["tmp_path"] / "validation.jsonl",
        [
            _formal_row(
                dataset="mulberry",
                original_id="mulberry-1",
                row_index=1,
                split="validation",
            )
        ],
    )

    rows, stats = normalize_stage(
        formal_source=staged["formal_path"],
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10},
    )

    assert rows == []
    assert stats.review_required == {"formal_split_is_not_train": 1}


def test_missing_image_is_review_required(staged: dict) -> None:
    _write_json_array(
        staged["raw_root"] / "mulberry-proxy" / "mulberry_sft.json",
        [_mulberry_record(image="cauldron/clevr/images/absent.png")],
    )
    formal_path = _write_jsonl(
        staged["tmp_path"] / "missing_image.jsonl",
        [
            _formal_row(
                dataset="mulberry",
                original_id="mulberry-260494",
                row_index=0,
                image_ref="cauldron/clevr/images/absent.png",
            )
        ],
    )

    rows, stats = normalize_stage(
        formal_source=formal_path,
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10},
    )

    assert rows == []
    assert stats.review_required == {"image_missing:0": 1}


def test_image_ref_disagreement_is_review_required(staged: dict) -> None:
    staged["formal_path"] = _write_jsonl(
        staged["tmp_path"] / "mismatch.jsonl",
        [
            _formal_row(
                dataset="mulberry",
                original_id="mulberry-1",
                row_index=1,
                image_ref="cauldron/clevr/images/other.png",
            )
        ],
    )

    rows, stats = normalize_stage(
        formal_source=staged["formal_path"],
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10},
    )

    assert rows == []
    assert stats.review_required == {"formal_image_ref_disagrees_with_raw_record": 1}


def test_unsupported_dataset_aborts_run(staged: dict) -> None:
    staged["formal_path"] = _write_jsonl(
        staged["tmp_path"] / "unsupported.jsonl",
        [_formal_row(dataset="thinklite", original_id="thinklite-1", row_index=0)],
    )

    with pytest.raises(UnsupportedDatasetError):
        normalize_stage(
            formal_source=staged["formal_path"],
            stage="stage1",
            usage_partition="sft_stage1",
            raw_root=staged["raw_root"],
            image_root=staged["image_root"],
            per_dataset_limit={"thinklite": 10},
        )


def test_per_dataset_limit_caps_rows(staged: dict) -> None:
    rows, stats = normalize_stage(
        formal_source=staged["formal_path"],
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 1, "retool": 1},
        raw_overrides={"retool": staged["retool_path"]},
    )

    assert [row.source_dataset for row in rows] == ["mulberry", "retool"]
    assert stats.dataset_rows_seen == {"mulberry": 1, "retool": 1}


def test_usage_partition_mismatch_is_review_required(staged: dict) -> None:
    staged["formal_path"] = _write_jsonl(
        staged["tmp_path"] / "wrong_stage.jsonl",
        [
            _formal_row(
                dataset="mulberry",
                original_id="mulberry-1",
                row_index=1,
                usage_partition="sft_stage2",
            )
        ],
    )

    rows, stats = normalize_stage(
        formal_source=staged["formal_path"],
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10},
    )

    assert rows == []
    assert stats.review_required == {"formal_usage_partition_mismatch": 1}


# --------------------------------------------------------------------------
# Source revision resolution
# --------------------------------------------------------------------------


def _null_revision_formal(
    staged: dict,
    name: str,
    *,
    dataset: str = "mulberry",
    original_id: str = "mulberry-260494",
    row_index: int = 1,
    image_ref: str | None = "cauldron/clevr/images/clevr_00011395.png",
) -> Path:
    return _write_jsonl(
        staged["tmp_path"] / name,
        [
            _formal_row(
                dataset=dataset,
                original_id=original_id,
                row_index=row_index,
                image_ref=image_ref,
                revision=None,
            )
        ],
    )


def test_declared_revision_origin_is_recorded(staged: dict) -> None:
    rows, stats = normalize_stage(
        formal_source=staged["formal_path"],
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10, "retool": 10},
        raw_overrides={"retool": staged["retool_path"]},
    )

    # A declared revision is passed through untouched.
    assert [row.source_revision for row in rows] == [REVISION, REVISION]
    assert [row.revision_origin for row in rows] == [
        REVISION_ORIGIN_DECLARED,
        REVISION_ORIGIN_DECLARED,
    ]
    assert stats.revision_origin_counts == {REVISION_ORIGIN_DECLARED: 2}


def test_missing_mulberry_revision_is_review_required(staged: dict) -> None:
    rows, stats = normalize_stage(
        formal_source=_null_revision_formal(staged, "mulberry-no-revision.jsonl"),
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"mulberry": 10},
    )

    # No revision is invented for a dataset that has no pinned fallback.
    assert rows == []
    assert stats.review_required == {"formal_source_revision_unresolvable": 1}


def test_missing_retool_revision_uses_the_pinned_constant(staged: dict) -> None:
    rows, stats = normalize_stage(
        formal_source=_null_revision_formal(
            staged,
            "retool-no-revision.jsonl",
            dataset="retool",
            original_id="retool-2",
            row_index=0,
            image_ref=None,
        ),
        stage="stage1",
        usage_partition="sft_stage1",
        raw_root=staged["raw_root"],
        image_root=staged["image_root"],
        per_dataset_limit={"retool": 10},
        raw_overrides={"retool": staged["retool_path"]},
    )

    assert stats.review_required == {}
    assert rows[0].source_revision == EXPECTED_RETOOL_REVISION
    assert rows[0].revision_origin == REVISION_ORIGIN_PINNED
    assert rows[0].to_dict()["revision_origin"] == REVISION_ORIGIN_PINNED
    assert stats.revision_origin_counts == {REVISION_ORIGIN_PINNED: 1}


# --------------------------------------------------------------------------
# Forbidden index
# --------------------------------------------------------------------------


def test_build_forbidden_rows_skips_unknown_revision_and_dedupes(
    tmp_path: Path,
) -> None:
    source = _write_jsonl(
        tmp_path / "test_split.jsonl",
        [
            {
                "record_id": "geometry3k:sft:aaa",
                "dataset": "geometry3k",
                "original_id": "2401",
                "resolved_revision": REVISION,
                "source_key": "a" * 64,
            },
            {
                "record_id": "geometry3k:sft:aaa",
                "dataset": "geometry3k",
                "original_id": "2401",
                "resolved_revision": REVISION,
                "source_key": "a" * 64,
            },
            {
                "record_id": "geometry3k:sft:bbb",
                "dataset": "geometry3k",
                "original_id": "2402",
                "resolved_revision": None,
                "source_key": "b" * 64,
            },
        ],
    )

    rows, report = build_forbidden_rows([source])

    assert len(rows) == 1
    assert rows[0]["task_id"] == "geometry3k:sft:aaa"
    assert rows[0]["source_content_hash"] == "a" * 64
    assert report.skipped == {
        "duplicate_identity": 1,
        "missing_resolved_revision": 1,
    }
    assert report.rows_written == 1
    assert report.files[0]["rows_seen"] == 3
    assert report.files[0]["rows_emitted"] == 1


def test_generated_forbidden_rows_load_into_forbidden_index(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "validation.jsonl",
        [
            {
                "record_id": "geometry3k:sft:ccc",
                "dataset": "geometry3k",
                "original_id": "2101",
                "resolved_revision": REVISION,
            }
        ],
    )
    rows, _ = build_forbidden_rows([source])
    forbidden_path = _write_jsonl(tmp_path / "forbidden_index_v1.jsonl", rows)

    index = ForbiddenSourceIndex.from_paths([forbidden_path])

    assert index.record_count == 1
