"""Strictly validate and concatenate the selected Stage-1 and Stage-2 SFT sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .dedup import record_hash
from .merge_sft import audit_record


def _has_repair_prompt(row: Dict[str, Any]) -> bool:
    return any(
        message.get("role") == "user"
        and "Now switch to the Self-Repair role." in message.get("content", "")
        for message in row.get("messages", [])
        if isinstance(message, dict)
    )


def _read_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield row


def _load_and_audit(path: Path, stage: int, *, allow_stage2_images: bool) -> List[Dict[str, Any]]:
    rows = list(_read_rows(path))
    if not rows:
        raise ValueError(f"No records found in {path}")

    failures: Dict[str, int] = {}
    seen: set[str] = set()
    for row_index, row in enumerate(rows, 1):
        reason = audit_record(
            row,
            stage,
            allow_stage2_images=allow_stage2_images,
        )
        if reason is not None:
            failures[reason] = failures.get(reason, 0) + 1
        digest = record_hash(row)
        if digest in seen:
            failures["duplicate_within_input"] = failures.get("duplicate_within_input", 0) + 1
        seen.add(digest)

    if failures:
        raise ValueError(f"Strict audit failed for Stage {stage} input {path}: {failures}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, chunks: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for chunk in chunks:
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _merge_rows(
    stage1_rows: Sequence[Dict[str, Any]],
    stage2_rows: Sequence[Dict[str, Any]],
    *,
    expected_stage1_rows: int,
    expected_stage2_rows: int,
    expected_stage1_repairs: int,
) -> List[Dict[str, Any]]:
    if len(stage1_rows) != expected_stage1_rows:
        raise ValueError(
            f"Stage 1 has {len(stage1_rows)} rows; expected {expected_stage1_rows}"
        )
    if len(stage2_rows) != expected_stage2_rows:
        raise ValueError(
            f"Stage 2 has {len(stage2_rows)} rows; expected {expected_stage2_rows}"
        )

    repair_rows = sum(_has_repair_prompt(row) for row in stage1_rows)
    if repair_rows != expected_stage1_repairs:
        raise ValueError(
            f"Stage 1 has {repair_rows} Repair rows; expected {expected_stage1_repairs}"
        )

    seen: set[str] = set()
    for stage_name, rows in (("Stage 1", stage1_rows), ("Stage 2", stage2_rows)):
        for row_index, row in enumerate(rows, 1):
            digest = record_hash(row)
            if digest in seen:
                raise ValueError(
                    f"Exact duplicate found at {stage_name} row {row_index}; "
                    "refusing to silently drop training data"
                )
            seen.add(digest)
    return [*stage1_rows, *stage2_rows]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--stage2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-stage1-rows", type=int, default=500)
    parser.add_argument("--expected-stage2-rows", type=int, default=500)
    parser.add_argument("--expected-stage1-repairs", type=int, default=6)
    parser.add_argument("--allow-stage2-images", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    stage1_path = args.stage1.resolve(strict=True)
    stage2_path = args.stage2.resolve(strict=True)
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if output_path in {stage1_path, stage2_path} or manifest_path in {stage1_path, stage2_path}:
        raise ValueError("Output and manifest paths must not overwrite either input dataset")
    if output_path == manifest_path:
        raise ValueError("Output dataset and manifest must use different paths")

    stage1_rows = _load_and_audit(stage1_path, 1, allow_stage2_images=False)
    stage2_rows = _load_and_audit(
        stage2_path,
        2,
        allow_stage2_images=args.allow_stage2_images,
    )
    merged = _merge_rows(
        stage1_rows,
        stage2_rows,
        expected_stage1_rows=args.expected_stage1_rows,
        expected_stage2_rows=args.expected_stage2_rows,
        expected_stage1_repairs=args.expected_stage1_repairs,
    )

    _atomic_write(
        output_path,
        (json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "concatenate_stage1_then_stage2_for_single_mixed_sft_run",
        "curriculum_note": "This mixes both stages in one run; it is not sequential Stage-1-to-Stage-2 annealing.",
        "stage1_input": str(stage1_path),
        "stage2_input": str(stage2_path),
        "stage1_sha256": _sha256(stage1_path),
        "stage2_sha256": _sha256(stage2_path),
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(stage2_rows),
        "stage1_repair_rows": sum(_has_repair_prompt(row) for row in stage1_rows),
        "total_rows": len(merged),
        "cross_and_within_input_duplicates": 0,
        "stage2_images_allowed": args.allow_stage2_images,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    _atomic_write(
        manifest_path,
        [json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
