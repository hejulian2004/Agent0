"""Validate an SFT JSONL file without creating or modifying a dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .dedup import record_hash
from .merge_sft import audit_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument(
        "--allow-stage2-images",
        action="store_true",
        help="Allow and validate local/URL images in multimodal Stage-2 rows",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    total_rows = 0
    valid_rows = 0
    duplicate_rows = 0
    rejected: Dict[str, int] = {}
    examples: Dict[str, int] = {}
    seen: set[str] = set()

    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                row: Any = json.loads(line)
            except json.JSONDecodeError:
                reason = "invalid_json"
            else:
                reason = audit_record(
                    row,
                    args.stage,
                    allow_stage2_images=args.allow_stage2_images,
                )
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                examples.setdefault(reason, line_number)
                continue
            digest = record_hash(row)
            if digest in seen:
                duplicate_rows += 1
                examples.setdefault("duplicate", line_number)
                continue
            seen.add(digest)
            valid_rows += 1

    expected_rows_ok = args.expected_rows is None or total_rows == args.expected_rows
    report = {
        "stage": args.stage,
        "input": str(args.input),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "duplicate_rows": duplicate_rows,
        "rejected": rejected,
        "first_problem_lines": examples,
        "expected_rows": args.expected_rows,
        "ok": not rejected and duplicate_rows == 0 and expected_rows_ok,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
