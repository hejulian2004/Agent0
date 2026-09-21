"""Select a small, strictly audited SFT validation subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from .dedup import record_hash
from .merge_sft import audit_record


def _read_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.count <= 0:
        raise ValueError("--count must be positive")

    selected: list[Dict[str, Any]] = []
    selected_hashes: set[str] = set()
    rejected: Dict[str, int] = {}
    input_rows = 0
    duplicate_rows = 0

    for row in _read_rows(args.input):
        input_rows += 1
        reason = audit_record(row, args.stage)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        digest = record_hash(row)
        if digest in selected_hashes:
            duplicate_rows += 1
            continue
        selected_hashes.add(digest)
        if len(selected) < args.count:
            selected.append(row)

    if len(selected) != args.count:
        raise RuntimeError(
            f"Only found {len(selected)} strictly valid Stage-{args.stage} rows; "
            f"required {args.count}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
    manifest = {
        "stage": args.stage,
        "input": str(args.input),
        "output": str(args.output),
        "input_rows": input_rows,
        "selected_rows": len(selected),
        "strict_audit": True,
        "duplicate_rows": duplicate_rows,
        "rejected": rejected,
        "selected_hashes": [record_hash(row) for row in selected],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
