"""Build RLHFDataset-compatible task rows from the bounded SFT smoke selection.

The paper's RL input is a task row, not a pre-generated group of trajectories.
This exporter therefore writes one row per task.  ``n=8`` is deliberately not
encoded here; the rollout expands each prompt at runtime.

This module is intentionally independent of the Teacher backend.  It reuses
the already selected/validated source rows from the SFT smoke run so that the
SFT and RL smoke sets refer to the same task identity and image bytes, while
rendering a clean RL prompt and using the task ground truth only as reward
metadata.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from tools.data_builder.schema import (
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    canonical_json,
)


RL_DATASETS: tuple[str, ...] = (
    "mathverse",
    "mathvista",
    "wemath",
    "arxivqa",
    "chartqa",
    "thinklite",
)

EVAL_SPLITS = {
    "test",
    "testmini",
    "test_public",
    "test_private",
    "validation",
    "val",
    "dev",
}


class RLSmokeExportError(ValueError):
    """Raised when the input smoke records cannot form valid RL rows."""


def _load_canonical_prompt_module() -> Any:
    """Load prompt definitions without importing the heavyweight verl package."""

    module_path = Path(__file__).resolve().parents[3] / "verl" / "prompts" / "agent0_templates.py"
    spec = importlib.util.spec_from_file_location("agent0vl_rl_prompt", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load canonical prompt module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PROMPTS = _load_canonical_prompt_module()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RLSmokeExportError(f"line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def _question_from_sft_record(record: Mapping[str, Any]) -> str:
    messages = record.get("messages") or []
    if not messages or not isinstance(messages[0], Mapping):
        raise RLSmokeExportError(f"{record.get('record_id')}: missing source user message")
    content = str(messages[0].get("content") or "")
    # The local SFT smoke prompt contains an outer ``## Question`` section
    # with generation instructions and a nested ``Question:`` section with
    # the actual source task.  Use the final explicit Question marker rather
    # than accidentally training RL on the smoke-generation instructions.
    matches = list(re.finditer(r"(?:^|\n)Question:[ \t]*\n", content))
    if matches:
        question = content[matches[-1].end():]
    else:
        marker = "## Question\n"
        start = content.find(marker)
        if start < 0:
            raise RLSmokeExportError(f"{record.get('record_id')}: source question marker missing")
        question = content[start + len(marker):]
    end_marker = "\n\nBegin the next Solver reasoning step."
    if end_marker in question:
        question = question.split(end_marker, 1)[0]
    question = question.strip()
    return question


def _load_mathverse_question_fallbacks(input_jsonl: Path) -> dict[str, str]:
    """Recover OCR-backed MathVerse questions missing from the source field.

    A few MathVerse rows intentionally have an empty ``question`` field while
    retaining the human-readable ``question_for_eval`` field.  The original
    SFT smoke run selected one of those rows, so the RL exporter uses the
    source Parquet only as a deterministic fallback for that specific case.
    """

    repo_root = input_jsonl.resolve().parents[3]
    parquet_path = repo_root / "data" / "raw" / ".staging" / "mathverse-proxy" / "testmini.parquet"
    if not parquet_path.is_file():
        return {}
    fallbacks: dict[str, str] = {}
    try:
        parquet = pq.ParquetFile(parquet_path)
        columns = ["sample_index", "problem_index", "question", "question_for_eval", "query_wo"]
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                problem_index = str(row.get("problem_index") or "")
                sample_index = str(row.get("sample_index") or "")
                key = f"{problem_index}-{sample_index}"
                question = str(row.get("question") or "").strip()
                question = question or str(row.get("question_for_eval") or "").strip()
                question = question or str(row.get("query_wo") or "").strip()
                if key and question:
                    fallbacks[key] = question
    except Exception:
        # The fallback is only needed for the smoke artifact.  A missing or
        # unreadable raw source must still fail later rather than silently
        # producing an empty RL prompt.
        return {}
    return fallbacks


def _score_type(value: Any) -> str:
    value = str(value or "exact").strip().lower()
    aliases = {
        "numeric": "exact",
        "number": "exact",
        "multiple_choice": "choice",
        "mcq": "choice",
    }
    value = aliases.get(value, value)
    if value not in {"math", "exact", "choice", "list"}:
        return "exact"
    return value


def _safe_relative_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RLSmokeExportError(f"image path is not relative to smoke root: {value!r}")
    path = (root / relative).resolve()
    root_resolved = root.resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as exc:
        raise RLSmokeExportError(f"image path escapes smoke root: {value!r}") from exc
    if not path.is_file():
        raise RLSmokeExportError(f"image file is missing: {path}")
    return path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_eval_split(split: str) -> bool:
    normalized = str(split or "").strip().lower()
    return normalized in EVAL_SPLITS or normalized.startswith("test")


def _formal_training_eligible(task: Mapping[str, Any]) -> bool:
    """Apply the smoke-level conservative eligibility gate.

    Phase 1.5 verification records are not available yet, so the smoke rows
    are never claimed to be formal-training eligible.  This helper still
    exposes the split/license portion of the gate for auditability.
    """

    split_allowed = not _is_eval_split(str(task.get("official_split") or ""))
    license_verified = str(task.get("license_status") or "unknown") == "verified"
    return bool(split_allowed and license_verified and task.get("verification_status") == "passed")


def _image_item(root: Path, relative_path: str, expected_hash: str | None) -> dict[str, Any]:
    path = _safe_relative_path(root, relative_path)
    image_bytes = path.read_bytes()
    actual_hash = _sha256_bytes(image_bytes)
    if expected_hash and actual_hash != expected_hash:
        raise RLSmokeExportError(
            f"image hash mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
        )
    # ``path`` intentionally remains null.  RL input must carry binary image
    # content and must not depend on a Windows-only local path.
    return {"bytes": image_bytes, "path": None}


def _capability_ability(task: Mapping[str, Any], has_image: bool) -> str:
    labels = {str(value) for value in task.get("capability_labels") or []}
    if has_image or "image_manipulation" in labels:
        return "visual_reasoning"
    if "math_code" in labels:
        return "math_reasoning"
    return "reasoning"


def _row_from_record(
    record: Mapping[str, Any],
    *,
    smoke_root: Path,
    index: int,
    question_fallbacks: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], bool]:
    metadata = record.get("metadata") or {}
    task = metadata.get("task") or {}
    if not isinstance(task, Mapping):
        raise RLSmokeExportError(f"{record.get('record_id')}: task metadata is missing")

    source = str(task.get("source") or "")
    task_id = str(task.get("task_id") or "")
    if not source or not task_id:
        raise RLSmokeExportError(f"{record.get('record_id')}: source/task_id is missing")

    image_paths = [str(value) for value in (record.get("images") or []) if value]
    image_hash = (task.get("hashes") or {}).get("image_hash")
    image_items = [
        _image_item(smoke_root, value, image_hash if len(image_paths) == 1 else None)
        for value in image_paths
    ]
    question = _question_from_sft_record(record)
    if not question and source == "mathverse":
        question = str((question_fallbacks or {}).get(str(task.get("original_id") or ""), "")).strip()
    if not question:
        raise RLSmokeExportError(f"{record.get('record_id')}: source question is empty")
    official_split = str(task.get("official_split") or "unknown")
    license_status = str(task.get("license_status") or "unknown")
    verification_status = "not_built"
    score_type = _score_type(task.get("answer_type"))
    ground_truth = str(task.get("ground_truth") or "")
    accepted_answers = [str(value) for value in (task.get("accepted_answers") or [])]
    hashes = task.get("hashes") or {}
    formal_eligible = _formal_training_eligible({
        "official_split": official_split,
        "license_status": license_status,
        "verification_status": verification_status,
    })

    user_content = _PROMPTS.render_solver_request(
        question,
        image_context="one attached image" if image_items else "",
    )
    if image_items:
        user_content = "<image>\n" + user_content

    extra_info = {
        "index": index,
        "task_id": task_id,
        "source": source,
        "official_split": official_split,
        "original_id": str(task.get("original_id") or ""),
        "question_hash": str(hashes.get("question_hash") or ""),
        "image_hash": str(hashes.get("image_hash") or "") if image_items else None,
        "source_revision": str(task.get("source_revision")) if task.get("source_revision") is not None else None,
        "stage": str(record.get("stage") or ""),
        "record_id": str(record.get("record_id") or ""),
        "source_trajectory_id": str(record.get("source_trajectory_id") or ""),
        "license_status": license_status,
        "verification_status": verification_status,
        "formal_training_eligible": formal_eligible,
        "smoke_only": True,
        "source_metadata_json": canonical_json(task.get("source_metadata") or {}),
    }
    row: dict[str, Any] = {
        "data_source": source,
        "prompt": [
            {"role": "system", "content": _PROMPTS.render_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        "reward_model": {
            "style": "rule",
            "score_type": score_type,
            "ground_truth": ground_truth,
            "accepted_answers": accepted_answers,
        },
        "ability": _capability_ability(task, bool(image_items)),
        "extra_info": extra_info,
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "scorer_version": SCORER_VERSION,
    }
    if image_items:
        row["images"] = image_items
    return row, bool(image_items)


def _prompt_type() -> pa.DataType:
    return pa.list_(pa.struct([
        pa.field("role", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
    ]))


def _image_type() -> pa.DataType:
    return pa.list_(pa.struct([
        pa.field("bytes", pa.binary(), nullable=False),
        pa.field("path", pa.string(), nullable=True),
    ]))


def _reward_type() -> pa.DataType:
    return pa.struct([
        pa.field("style", pa.string(), nullable=False),
        pa.field("score_type", pa.string(), nullable=False),
        pa.field("ground_truth", pa.string(), nullable=False),
        pa.field("accepted_answers", pa.list_(pa.string()), nullable=False),
    ])


def _extra_info_type() -> pa.DataType:
    return pa.struct([
        pa.field("index", pa.int64(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("official_split", pa.string(), nullable=False),
        pa.field("original_id", pa.string(), nullable=False),
        pa.field("question_hash", pa.string(), nullable=False),
        pa.field("image_hash", pa.string(), nullable=True),
        pa.field("source_revision", pa.string(), nullable=True),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("source_trajectory_id", pa.string(), nullable=False),
        pa.field("license_status", pa.string(), nullable=False),
        pa.field("verification_status", pa.string(), nullable=False),
        pa.field("formal_training_eligible", pa.bool_(), nullable=False),
        pa.field("smoke_only", pa.bool_(), nullable=False),
        pa.field("source_metadata_json", pa.string(), nullable=False),
    ])


def _schema(*, multimodal: bool) -> pa.Schema:
    fields = [
        pa.field("data_source", pa.string(), nullable=False),
        pa.field("prompt", _prompt_type(), nullable=False),
    ]
    if multimodal:
        fields.append(pa.field("images", _image_type(), nullable=False))
    fields.extend([
        pa.field("reward_model", _reward_type(), nullable=False),
        pa.field("ability", pa.string(), nullable=False),
        pa.field("extra_info", _extra_info_type(), nullable=False),
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("protocol_version", pa.string(), nullable=False),
        pa.field("scorer_version", pa.string(), nullable=False),
    ])
    return pa.schema(fields)


def build_rl_smoke_rows(
    input_jsonl: Path,
    *,
    samples_per_dataset: int = 10,
    datasets: Sequence[str] = RL_DATASETS,
    allow_eval_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic RL rows from the existing SFT smoke records."""

    if samples_per_dataset <= 0:
        raise RLSmokeExportError("samples_per_dataset must be positive")
    requested = tuple(str(value) for value in datasets)
    unknown = sorted(set(requested) - set(RL_DATASETS))
    if unknown:
        raise RLSmokeExportError(f"unsupported RL smoke datasets: {unknown}")

    smoke_root = input_jsonl.parent.resolve()
    source_records = _read_jsonl(input_jsonl)
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in requested}
    for record in source_records:
        task = ((record.get("metadata") or {}).get("task") or {})
        source = str(task.get("source") or "")
        if source in grouped:
            grouped[source].append(record)

    rows: list[dict[str, Any]] = []
    question_fallbacks = _load_mathverse_question_fallbacks(input_jsonl) if "mathverse" in requested else {}
    counts: Counter[str] = Counter()
    blocked_eval: list[dict[str, Any]] = []
    blocked_license: list[dict[str, Any]] = []
    source_selection: list[dict[str, Any]] = []
    for dataset in requested:
        selected = grouped[dataset][:samples_per_dataset]
        if len(selected) != samples_per_dataset:
            raise RLSmokeExportError(
                f"{dataset}: expected {samples_per_dataset} source records, found {len(selected)}"
            )
        for record in selected:
            task = ((record.get("metadata") or {}).get("task") or {})
            split = str(task.get("official_split") or "unknown")
            if _is_eval_split(split) and not allow_eval_only:
                raise RLSmokeExportError(
                    f"{dataset}: {split} is eval-only; pass --allow-eval-only for debug smoke output"
                )
            row, multimodal = _row_from_record(
                record,
                smoke_root=smoke_root,
                index=len(rows),
                question_fallbacks=question_fallbacks,
            )
            rows.append(row)
            counts[dataset] += 1
            task_info = row["extra_info"]
            source_selection.append({
                "record_id": task_info["record_id"],
                "task_id": task_info["task_id"],
                "dataset": dataset,
                "official_split": split,
                "multimodal": multimodal,
                "image_hash": task_info["image_hash"],
                "question_hash": task_info["question_hash"],
                "license_status": task_info["license_status"],
                "verification_status": task_info["verification_status"],
                "formal_training_eligible": task_info["formal_training_eligible"],
            })
            if _is_eval_split(split):
                blocked_eval.append({"dataset": dataset, "task_id": task_info["task_id"], "split": split})
            if task_info["license_status"] != "verified":
                blocked_license.append({
                    "dataset": dataset,
                    "task_id": task_info["task_id"],
                    "license_status": task_info["license_status"],
                })

    if len(rows) != len(requested) * samples_per_dataset:
        raise RLSmokeExportError("RL smoke row count does not match the requested per-dataset count")
    record_ids = [row["extra_info"]["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise RLSmokeExportError("duplicate source record IDs in RL smoke rows")

    report = {
        "build_mode": "rl_task_row_smoke_from_sft_selection",
        "paper_scope": "auditable reconstruction; not the undisclosed paper RL mixture",
        "datasets": list(requested),
        "requested_samples_per_dataset": samples_per_dataset,
        "rl_task_row_count": len(rows),
        "n_repeat": "runtime_only; not stored in parquet",
        "multimodal_row_count": sum("images" in row for row in rows),
        "text_only_row_count": sum("images" not in row for row in rows),
        "dataset_counts": dict(counts),
        "eval_only_debug_row_count": len(blocked_eval),
        "eval_only_debug_rows": blocked_eval,
        "license_blocked_row_count": len(blocked_license),
        "license_blocked_rows": blocked_license,
        "formal_training_eligible_row_count": sum(
            bool(row["extra_info"]["formal_training_eligible"]) for row in rows
        ),
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "scorer_version": SCORER_VERSION,
        "source_selection": source_selection,
        "notes": [
            "Rows are task inputs; the rollout creates n=8 generations at runtime.",
            "Images are stored as binary bytes in multimodal Parquet; no Windows path is required by RLHFDataset.",
            "Eval-only rows are present only because this is an explicitly debug smoke build.",
            "Formal eligibility remains false until a Phase 1.5 verification record passes split and license gates.",
        ],
    }
    return rows, report


def _write_table(rows: Sequence[Mapping[str, Any]], path: Path, *, multimodal: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _schema(multimodal=multimodal)
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table, path, compression="zstd")


def write_rl_smoke_parquet(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Write separate multimodal/text-only Parquet files and return paths/counts."""

    multimodal_rows = [row for row in rows if "images" in row]
    text_only_rows = [row for row in rows if "images" not in row]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {
        "multimodal": None,
        "text_only": None,
        "multimodal_count": len(multimodal_rows),
        "text_only_count": len(text_only_rows),
    }
    if multimodal_rows:
        path = output_dir / "train_multimodal.parquet"
        _write_table(multimodal_rows, path, multimodal=True)
        outputs["multimodal"] = str(path)
    if text_only_rows:
        path = output_dir / "train_text_only.parquet"
        _write_table(text_only_rows, path, multimodal=False)
        outputs["text_only"] = str(path)
    return outputs
