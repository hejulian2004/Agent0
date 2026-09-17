"""Normalize the frozen formal SFT partition into the local source contract.

Phase 2A consumes ``agent0vl.local_sft_builder.source.v1`` rows, but the formal
partition files only carry source *identity* plus ``raw_path``/``row_index``/
``image_ref``.  They do not carry ``question`` or ``ground_truth``.  This module
materializes those fields from the raw dataset artifacts and emits normalized
source JSONL that ``RealSourceAdapter`` accepts unchanged.

Design constraints:

* Only ``formal_sft_stage{1,2}.jsonl`` is read.  Those rows are train-only and
  ``license_status=verified``; every row is re-checked here anyway.
* The ReTool rows are text-only.  Mulberry rows are single-image.  A single
  ``image_root`` therefore covers every image in one normalized file, which is
  what ``RealSourceAdapter`` requires.
* ``ground_truth`` is a *reference label* taken from the source artifact's own
  assistant message.  It is recorded with
  ``ground_truth_origin=source_assistant_final_answer`` and must never be
  presented as independent correctness verification.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .canonical import normalize_text, sha256_file, sha256_json

SOURCE_SCHEMA_VERSION = "agent0vl.local_sft_builder.source.v1"
NORMALIZE_VERSION = "agent0vl.local_sft_builder.source_normalize.v1"

GROUND_TRUTH_ORIGIN = "source_assistant_final_answer"
ANSWER_CHECK_METHOD = "reference_answer_match_v1"

_IMAGE_TOKEN = "<image>"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FINAL_ANSWER_RE = re.compile(
    r"the final answer is\s*:?\s*([^\n]*)",
    re.IGNORECASE,
)
_RETOOL_QUESTION_RE = re.compile(r"\*\*user question:\*\*\s*", re.IGNORECASE)

# The real ReTool rows end with ``<answer>\boxed{...}</answer>`` and everything
# before that is reasoning, so the raw assistant text must never be used as the
# reference label.  The order follows ``agent0_evaluator._extract_answer``: the
# boxed value first, then the ``<answer>`` block.
_RETOOL_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_RETOOL_ANSWER_BLOCK_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE
)

# Raw artifact locations relative to the staging root.  These are the same on
# Windows and on vcc, so the normalized source stays machine independent.
_DEFAULT_RAW_MEMBERS: dict[str, str] = {
    "mulberry": "mulberry-proxy/mulberry_sft.json",
    "retool": "retool-proxy/train_2000.parquet",
}

# ``source_revision`` provenance.
#
# Mulberry rows carry a usable ``resolved_revision`` in the frozen partition and
# are recorded as declared.  ReTool rows carry ``resolved_revision: null``, so
# Phase 2B-Lite pins their revision to a manually verified upstream commit and
# labels it as pinned -- never as declared, and deliberately never as derived
# from whichever local git checkout happens to be present.  Reading a local HEAD
# proves nothing about the provenance of ``train_2000.parquet``.
#
# Phase 2B-Lite does NOT verify this constant against the remote or the LFS
# pointer.  It is a data-version label for this round only.
REVISION_ORIGIN_DECLARED = "declared_resolved_revision"
REVISION_ORIGIN_PINNED = "pinned_retool_revision"

EXPECTED_RETOOL_REVISION = "13eb7a396284caa114d677af3d071864c27ba5cc"


class SourceNormalizeError(RuntimeError):
    """Base class for failures that invalidate the whole normalization run."""


class RawSourceError(SourceNormalizeError):
    """Raised when a raw dataset artifact cannot be read or parsed."""


class UnsupportedDatasetError(SourceNormalizeError):
    """Raised when a formal row names a dataset with no adapter."""


class RowReviewRequired(ValueError):
    """A data-quality issue that affects one row but not the whole run."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


# --------------------------------------------------------------------------
# Streaming JSON array reader
# --------------------------------------------------------------------------


def iter_json_array(path: Path, *, read_size: int = 1 << 20) -> Iterator[Any]:
    """Yield the elements of a top-level JSON array without loading it whole.

    ``mulberry_sft.json`` is a 486 MB array, so ``json.load`` is not an option.
    Elements are decoded incrementally with ``JSONDecoder.raw_decode`` against a
    sliding buffer.
    """

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        while True:
            if position >= len(buffer):
                chunk = handle.read(read_size)
                if not chunk:
                    if not started:
                        raise RawSourceError(
                            f"{path} is empty and cannot be a JSON array"
                        )
                    if buffer[position:].strip() not in ("", "]", ","):
                        raise RawSourceError(f"{path} has an unterminated JSON array")
                    return
                buffer = buffer[position:] + chunk
                position = 0
                continue

            if not started:
                head = next(
                    (index for index in range(position, len(buffer)) if not buffer[index].isspace()),
                    None,
                )
                if head is None:
                    position = len(buffer)
                    continue
                if buffer[head] != "[":
                    raise RawSourceError(f"{path} does not start with a JSON array")
                position = head + 1
                started = True
                continue

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position >= len(buffer):
                continue
            if buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except ValueError:
                chunk = handle.read(read_size)
                if not chunk:
                    raise RawSourceError(
                        f"{path} contains a malformed JSON element near offset {position}"
                    ) from None
                buffer = buffer[position:] + chunk
                position = 0
                continue
            yield value
            position = end


# --------------------------------------------------------------------------
# Per-dataset extraction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedSource:
    question: str
    ground_truth: str
    images: tuple[str, ...]


def _message_contents(record: Mapping[str, Any], dataset: str) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RowReviewRequired(f"{dataset}_record_missing_messages")
    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise RowReviewRequired(f"{dataset}_message_not_mapping")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise RowReviewRequired(f"{dataset}_message_missing_role_or_content")
        normalized.append({"role": role, "content": content})
    return normalized


def _clean_reference_answer(value: str) -> str:
    text = value.strip()
    text = text.rstrip("#").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def extract_mulberry_final_answer(assistant_text: str) -> str | None:
    """Return the last ``The final answer is:`` value, or ``None``."""

    matches = _FINAL_ANSWER_RE.findall(assistant_text)
    if not matches:
        return None
    candidate = _clean_reference_answer(matches[-1])
    return candidate or None


def extract_retool_final_answer(assistant_text: str) -> str | None:
    """Return the ReTool reference answer, never the raw reasoning text.

    The real source rows end with ``<answer>\\boxed{...}</answer>``; the whole
    assistant message is the reference trajectory's reasoning, so it cannot be
    used as the reference label.  Taking it verbatim made every ReTool row fail
    the reference-answer check and therefore the export gate.
    """

    boxed = _RETOOL_BOXED_RE.findall(assistant_text)
    if boxed:
        candidate = _clean_reference_answer(boxed[-1])
        if candidate:
            return candidate
    blocks = _RETOOL_ANSWER_BLOCK_RE.findall(assistant_text)
    if blocks:
        candidate = _clean_reference_answer(blocks[-1])
        if candidate:
            return candidate
    return None


def extract_retool_question(user_text: str) -> str | None:
    matches = list(_RETOOL_QUESTION_RE.finditer(user_text))
    if not matches:
        return None
    candidate = user_text[matches[-1].end():].strip()
    return candidate or None


def _normalize_image_field(value: Any, dataset: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise RowReviewRequired(f"{dataset}_images_must_be_string_or_list")
    images: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise RowReviewRequired(f"{dataset}_image_path_must_be_non_empty_string")
        images.append(normalize_text(item).strip().replace("\\", "/"))
    return tuple(images)


def extract_mulberry(record: Mapping[str, Any]) -> ExtractedSource:
    messages = _message_contents(record, "mulberry")
    question = messages[0]["content"]
    assistant = next(
        (item["content"] for item in reversed(messages) if item["role"] == "assistant"),
        None,
    )
    if assistant is None:
        raise RowReviewRequired("mulberry_record_missing_assistant_message")
    ground_truth = extract_mulberry_final_answer(assistant)
    if ground_truth is None:
        raise RowReviewRequired("mulberry_reference_answer_not_parseable")
    images = _normalize_image_field(record.get("images"), "mulberry")
    return ExtractedSource(
        question=normalize_text(question),
        ground_truth=normalize_text(ground_truth),
        images=images,
    )


def extract_retool(record: Mapping[str, Any]) -> ExtractedSource:
    messages = _message_contents(record, "retool")
    user_text = next(
        (item["content"] for item in reversed(messages) if item["role"] == "user"),
        None,
    )
    if user_text is None:
        raise RowReviewRequired("retool_record_missing_user_message")
    question = extract_retool_question(user_text)
    if question is None:
        raise RowReviewRequired("retool_question_marker_not_found")
    assistant = next(
        (item["content"] for item in reversed(messages) if item["role"] == "assistant"),
        None,
    )
    if assistant is None:
        raise RowReviewRequired("retool_record_missing_assistant_message")
    ground_truth = extract_retool_final_answer(assistant)
    if not ground_truth:
        raise RowReviewRequired("retool_reference_answer_not_parseable")
    # ReTool is text-only: this is what keeps the single image_root invariant.
    return ExtractedSource(
        question=normalize_text(question),
        ground_truth=normalize_text(ground_truth),
        images=(),
    )


_EXTRACTORS: dict[str, Callable[[Mapping[str, Any]], ExtractedSource]] = {
    "mulberry": extract_mulberry,
    "retool": extract_retool,
}


# --------------------------------------------------------------------------
# Raw record loading
# --------------------------------------------------------------------------


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on target env
        raise RawSourceError(
            "pyarrow is required to read the ReTool parquet source"
        ) from exc
    table = parquet.read_table(str(path))
    return [dict(row) for row in table.to_pylist()]


def resolve_source_revision(
    formal_row: Mapping[str, Any],
    dataset: str,
) -> tuple[str, str]:
    """Return ``(source_revision, revision_origin)`` for one formal row.

    A usable ``resolved_revision`` is taken as declared.  ReTool rows carry none
    in the frozen partition, so they fall back to the revision pinned for this
    round.  Any other dataset without one goes to review: a revision is never
    invented for it.
    """

    declared = formal_row.get("resolved_revision")
    if isinstance(declared, str):
        candidate = normalize_text(declared).strip()
        if candidate and candidate.casefold() != "unknown":
            return candidate, REVISION_ORIGIN_DECLARED
    if dataset == "retool":
        return EXPECTED_RETOOL_REVISION, REVISION_ORIGIN_PINNED
    raise RowReviewRequired("formal_source_revision_unresolvable")


def fetch_raw_records(
    dataset: str,
    raw_path: Path,
    wanted: Iterable[int],
) -> dict[int, Mapping[str, Any]]:
    """Return ``{row_index: raw record}`` for the requested indices only."""

    targets = set(wanted)
    if not targets:
        return {}
    found: dict[int, Mapping[str, Any]] = {}

    if raw_path.suffix.lower() == ".parquet":
        rows = _load_parquet_rows(raw_path)
        for index in sorted(targets):
            if 0 <= index < len(rows):
                found[index] = rows[index]
        return found

    for index, record in enumerate(iter_json_array(raw_path)):
        if index in targets:
            if not isinstance(record, Mapping):
                raise RawSourceError(
                    f"{raw_path} element {index} is not a JSON object"
                )
            found[index] = record
            if len(found) == len(targets):
                break
    return found


# --------------------------------------------------------------------------
# Row normalization
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedRow:
    task_id: str
    source_record_id: str
    original_id: str
    source_dataset: str
    source_revision: str
    split: str
    usage_partition: str
    question: str
    ground_truth: str
    images: tuple[str, ...]
    image_refs: tuple[dict[str, str], ...]
    revision_origin: str = REVISION_ORIGIN_DECLARED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "task_id": self.task_id,
            "source_record_id": self.source_record_id,
            "original_id": self.original_id,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "revision_origin": self.revision_origin,
            "split": self.split,
            "usage_partition": self.usage_partition,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "ground_truth_origin": GROUND_TRUTH_ORIGIN,
            "answer_check_method": ANSWER_CHECK_METHOD,
            "images": list(self.images),
            "image_refs": [dict(ref) for ref in self.image_refs],
            "image_count": len(self.images),
        }


@dataclass
class NormalizeStats:
    dataset_rows_seen: dict[str, int] = field(default_factory=dict)
    written: int = 0
    review_required: dict[str, int] = field(default_factory=dict)
    review_details: list[dict[str, Any]] = field(default_factory=list)
    revision_origin_counts: dict[str, int] = field(default_factory=dict)

    def note_revision_origin(self, origin: str) -> None:
        self.revision_origin_counts[origin] = (
            self.revision_origin_counts.get(origin, 0) + 1
        )

    def note_review(self, dataset: str, locator: int, code: str) -> None:
        """Record one skipped row.

        ``locator`` is the raw ``row_index`` when it is known, and the formal
        file line number when the row could not be indexed at all.
        """

        self.review_required[code] = self.review_required.get(code, 0) + 1
        self.review_details.append(
            {"dataset": dataset, "location": locator, "reason": code}
        )


def _require_text(row: Mapping[str, Any], field_name: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str):
        raise RowReviewRequired(f"formal_{field_name}_must_be_string")
    text = normalize_text(value).strip()
    if not text:
        raise RowReviewRequired(f"formal_{field_name}_must_be_non_empty")
    return text


def _require_int(row: Mapping[str, Any], field_name: str) -> int:
    value = row.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RowReviewRequired(f"formal_{field_name}_must_be_int")
    if value < 0:
        raise RowReviewRequired(f"formal_{field_name}_must_be_non_negative")
    return value


def _resolve_image(image_root: Path, relative: str, index: int) -> tuple[Path, str]:
    if relative.startswith("/") or re.match(r"^[A-Za-z]:", relative):
        raise RowReviewRequired(f"absolute_image_path:{index}")
    if ".." in relative.split("/"):
        raise RowReviewRequired(f"image_path_traversal:{index}")
    candidate = (image_root / Path(relative)).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError as exc:
        raise RowReviewRequired(f"image_path_outside_root:{index}") from exc
    if not candidate.is_file():
        raise RowReviewRequired(f"image_missing:{index}")
    return candidate, sha256_file(candidate)


def normalize_formal_row(
    formal_row: Mapping[str, Any],
    raw_record: Mapping[str, Any],
    *,
    dataset: str,
    stage: str,
    usage_partition: str,
    task_id: str,
    image_root: Path | None,
) -> NormalizedRow:
    """Build one normalized source row, or raise ``RowReviewRequired``."""

    source_record_id = _require_text(formal_row, "source_record_id")
    original_id = _require_text(formal_row, "original_id")
    source_revision, revision_origin = resolve_source_revision(formal_row, dataset)
    split = _require_text(formal_row, "split")
    if split != "train":
        raise RowReviewRequired("formal_split_is_not_train")
    declared_partition = _require_text(formal_row, "usage_partition")
    if declared_partition != usage_partition:
        raise RowReviewRequired("formal_usage_partition_mismatch")
    if formal_row.get("license_status") != "verified":
        raise RowReviewRequired("formal_license_not_verified")
    if formal_row.get("formal_training_eligible") is not True:
        raise RowReviewRequired("formal_row_not_training_eligible")

    extractor = _EXTRACTORS.get(dataset)
    if extractor is None:
        raise UnsupportedDatasetError(f"no adapter for dataset {dataset!r}")
    extracted = extractor(raw_record)

    declared_image = formal_row.get("image_ref")
    if isinstance(declared_image, str) and declared_image.strip():
        declared = normalize_text(declared_image).strip().replace("\\", "/")
        if not extracted.images:
            raise RowReviewRequired("formal_image_ref_without_raw_image")
        if declared not in extracted.images:
            raise RowReviewRequired("formal_image_ref_disagrees_with_raw_record")

    if extracted.question.count(_IMAGE_TOKEN) != len(extracted.images):
        raise RowReviewRequired("image_placeholder_count_mismatch")

    if extracted.images:
        if image_root is None:
            raise RowReviewRequired("image_root_required_for_visual_source")
        refs: list[dict[str, str]] = []
        for index, relative in enumerate(extracted.images):
            _path, content_hash = _resolve_image(image_root, relative, index)
            refs.append(
                {
                    "asset_id": f"{dataset}/{original_id}/image-{index}",
                    "content_sha256": content_hash,
                }
            )
    else:
        refs = []

    return NormalizedRow(
        task_id=task_id,
        source_record_id=source_record_id,
        original_id=original_id,
        source_dataset=dataset,
        source_revision=source_revision,
        split=split,
        usage_partition=usage_partition,
        question=extracted.question,
        ground_truth=extracted.ground_truth,
        images=extracted.images,
        image_refs=tuple(refs),
        revision_origin=revision_origin,
    )


# --------------------------------------------------------------------------
# Forbidden identity index
# --------------------------------------------------------------------------


def _forbidden_content_hash(row: Mapping[str, Any]) -> str:
    source_key = row.get("source_key")
    if isinstance(source_key, str) and _SHA256_RE.fullmatch(source_key):
        return source_key
    return sha256_json(
        {
            "dataset": row.get("dataset"),
            "revision": row.get("resolved_revision"),
            "original_id": row.get("original_id"),
        }
    )


@dataclass
class ForbiddenIndexReport:
    """Coverage report for the leakage-prevention identity index.

    Rows whose frozen partition record carries no usable ``resolved_revision``
    cannot form a ``(dataset, revision, original_id)`` identity and are counted
    here rather than silently dropped.
    """

    rows_written: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_written": self.rows_written,
            "skipped": dict(self.skipped),
            "files": [dict(item) for item in self.files],
        }


def build_forbidden_rows(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], ForbiddenIndexReport]:
    """Project leakage-prevention sources onto the forbidden-index contract."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    report = ForbiddenIndexReport()

    def skip(code: str) -> None:
        report.skipped[code] = report.skipped.get(code, 0) + 1

    for path in paths:
        if not path.is_file():
            raise RawSourceError(f"forbidden source is not readable: {path}")
        rows_seen = 0
        rows_emitted = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_seen += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RawSourceError(f"malformed JSONL at {path}") from exc
                if not isinstance(row, Mapping):
                    raise RawSourceError(f"non-object row at {path}")
                dataset = row.get("dataset")
                original_id = row.get("original_id")
                revision = row.get("resolved_revision")
                if not isinstance(dataset, str) or not dataset.strip():
                    skip("missing_dataset")
                    continue
                if not isinstance(original_id, str) or not original_id.strip():
                    skip("missing_original_id")
                    continue
                if not isinstance(revision, str) or not revision.strip():
                    skip("missing_resolved_revision")
                    continue
                if revision.strip().casefold() == "unknown":
                    skip("unknown_resolved_revision")
                    continue
                identity = (dataset.strip(), revision.strip(), original_id.strip())
                if identity in seen:
                    skip("duplicate_identity")
                    continue
                seen.add(identity)
                task_id = row.get("record_id") or row.get("source_record_id")
                if not isinstance(task_id, str) or not task_id.strip():
                    task_id = f"forbidden:{dataset.strip()}:{original_id.strip()}"
                rows.append(
                    {
                        "task_id": task_id.strip(),
                        "source_dataset": dataset.strip(),
                        "source_revision": revision.strip(),
                        "original_id": original_id.strip(),
                        "source_content_hash": _forbidden_content_hash(row),
                    }
                )
                rows_emitted += 1
        report.files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows_seen": rows_seen,
                "rows_emitted": rows_emitted,
            }
        )
    report.rows_written = len(rows)
    return rows, report


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def _parse_per_dataset_limit(values: Sequence[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SourceNormalizeError(
                f"--per-dataset-limit expects dataset=count, received {value!r}"
            )
        dataset, raw_count = value.split("=", 1)
        dataset = dataset.strip()
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise SourceNormalizeError(
                f"--per-dataset-limit count must be an integer: {value!r}"
            ) from exc
        if not dataset or count <= 0:
            raise SourceNormalizeError(
                f"--per-dataset-limit needs a positive count: {value!r}"
            )
        limits[dataset] = count
    return limits


def normalize_stage(
    *,
    formal_source: Path,
    stage: str,
    usage_partition: str,
    raw_root: Path,
    image_root: Path | None,
    per_dataset_limit: Mapping[str, int],
    raw_overrides: Mapping[str, Path] | None = None,
) -> tuple[list[NormalizedRow], NormalizeStats]:
    """Normalize the head of one formal partition file."""

    if not formal_source.is_file():
        raise RawSourceError(f"formal source is not readable: {formal_source}")
    if not raw_root.is_dir():
        raise RawSourceError(f"raw root is not a directory: {raw_root}")

    overrides = dict(raw_overrides or {})
    stats = NormalizeStats()
    # dataset -> ordered list of (row_index, formal row)
    pending: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    exhausted: set[str] = set()

    with formal_source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RawSourceError(
                    f"malformed JSONL at {formal_source}:{line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise RawSourceError(f"non-object row at {formal_source}:{line_number}")
            dataset = row.get("dataset")
            if not isinstance(dataset, str) or not dataset.strip():
                raise SourceNormalizeError(
                    f"formal row without dataset at {formal_source}:{line_number}"
                )
            dataset = dataset.strip()
            if dataset in exhausted:
                continue
            if dataset not in _EXTRACTORS:
                raise UnsupportedDatasetError(
                    f"formal row names unsupported dataset {dataset!r} "
                    f"at {formal_source}:{line_number}"
                )
            limit = per_dataset_limit.get(dataset)
            bucket = pending.setdefault(dataset, [])
            if limit is not None and len(bucket) >= limit:
                exhausted.add(dataset)
                continue
            try:
                row_index = _require_int(row, "row_index")
            except RowReviewRequired as exc:
                stats.note_review(dataset, line_number, exc.code)
                continue
            bucket.append((row_index, row))
            stats.dataset_rows_seen[dataset] = stats.dataset_rows_seen.get(dataset, 0) + 1
            if limit is not None and all(
                len(pending.get(name, [])) >= per_dataset_limit.get(name, 0)
                for name in per_dataset_limit
                if name in _EXTRACTORS
            ):
                break

    ordered_datasets = sorted(pending)
    rows: list[NormalizedRow] = []
    sequence = 0
    for dataset in ordered_datasets:
        bucket = pending[dataset]
        if not bucket:
            continue
        member = overrides.get(dataset) or (raw_root / _DEFAULT_RAW_MEMBERS[dataset])
        if not member.is_file():
            raise RawSourceError(f"raw artifact for {dataset} is not readable: {member}")
        records = fetch_raw_records(dataset, member, [index for index, _ in bucket])
        for row_index, formal_row in bucket:
            raw_record = records.get(row_index)
            if raw_record is None:
                stats.note_review(dataset, row_index, "raw_row_index_not_found")
                continue
            # Only advance the sequence for rows that are actually emitted, so
            # task ids stay contiguous and stable for a given input.
            candidate_sequence = sequence + 1
            task_id = f"{dataset}-{stage}-{candidate_sequence:06d}"
            try:
                row = normalize_formal_row(
                    formal_row,
                    raw_record,
                    dataset=dataset,
                    stage=stage,
                    usage_partition=usage_partition,
                    task_id=task_id,
                    image_root=image_root,
                )
            except RowReviewRequired as exc:
                stats.note_review(dataset, row_index, exc.code)
                continue
            sequence = candidate_sequence
            stats.note_revision_origin(row.revision_origin)
            rows.append(row)
    stats.written = len(rows)
    return rows, stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize the formal SFT partition into the local source contract."
    )
    parser.add_argument("--formal-source", required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--forbidden-output")
    parser.add_argument(
        "--forbidden-source",
        action="append",
        default=[],
        help="leakage-prevention JSONL (test/validation/reserve/rl partitions)",
    )
    parser.add_argument(
        "--per-dataset-limit",
        action="append",
        default=[],
        help="dataset=count, repeatable (default: 2000 per dataset)",
    )
    parser.add_argument("--dataset-raw-path", action="append", default=[])
    parser.add_argument("--summary-output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        limits = _parse_per_dataset_limit(args.per_dataset_limit)
        if not limits:
            limits = {name: 2000 for name in _DEFAULT_RAW_MEMBERS}
        overrides: dict[str, Path] = {}
        for value in args.dataset_raw_path:
            if "=" not in value:
                raise SourceNormalizeError(
                    f"--dataset-raw-path expects dataset=path, received {value!r}"
                )
            dataset, raw_path = value.split("=", 1)
            overrides[dataset.strip()] = Path(raw_path).expanduser().resolve()

        usage_partition = f"sft_{args.stage}"
        rows, stats = normalize_stage(
            formal_source=Path(args.formal_source).expanduser().resolve(),
            stage=args.stage,
            usage_partition=usage_partition,
            raw_root=Path(args.raw_root).expanduser().resolve(),
            image_root=(
                Path(args.image_root).expanduser().resolve() if args.image_root else None
            ),
            per_dataset_limit=limits,
            raw_overrides=overrides,
        )

        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        written = _write_jsonl(output, (row.to_dict() for row in rows))

        summary: dict[str, Any] = {
            "normalize_version": NORMALIZE_VERSION,
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "stage": args.stage,
            "usage_partition": usage_partition,
            "formal_source": str(Path(args.formal_source).expanduser().resolve()),
            "formal_source_sha256": sha256_file(
                Path(args.formal_source).expanduser().resolve()
            ),
            "output": str(output),
            "rows_written": written,
            "dataset_rows_seen": stats.dataset_rows_seen,
            "review_required": stats.review_required,
            "revision_origin_counts": stats.revision_origin_counts,
            "ground_truth_origin": GROUND_TRUTH_ORIGIN,
            "answer_check_method": ANSWER_CHECK_METHOD,
            "independently_verified": False,
        }

        if args.forbidden_output:
            forbidden_rows, report = build_forbidden_rows(
                [Path(path).expanduser().resolve() for path in args.forbidden_source]
            )
            forbidden_path = Path(args.forbidden_output).expanduser().resolve()
            forbidden_path.parent.mkdir(parents=True, exist_ok=True)
            summary["forbidden_output"] = str(forbidden_path)
            summary["forbidden_rows"] = _write_jsonl(forbidden_path, forbidden_rows)
            summary["forbidden_report"] = report.to_dict()
            summary["forbidden_coverage_note"] = (
                "Rows whose frozen partition record has no usable "
                "resolved_revision cannot form a (dataset, revision, original_id) "
                "identity and are excluded from this index. They remain blocked "
                "structurally: only formal_sft_stage* rows are normalized, and "
                "SourceLeakageGuard rejects any split!=train or "
                "usage_partition outside {sft_stage1, sft_stage2}."
            )

        if args.summary_output:
            summary_path = Path(args.summary_output).expanduser().resolve()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except (OSError, SourceNormalizeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
