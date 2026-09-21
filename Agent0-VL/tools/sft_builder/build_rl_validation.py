"""Build a small verl-compatible multimodal RL validation Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    return parser


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.count <= 0:
        raise ValueError("--count must be positive")

    table = pq.read_table(
        args.input,
        columns=["ID", "split", "question", "option", "answer", "image_path"],
        use_threads=False,
    )
    rows: list[Dict[str, Any]] = []
    preview: list[Dict[str, Any]] = []
    for source_row in table.to_pylist():
        question = _clean_text(source_row.get("question"))
        option = _clean_text(source_row.get("option"))
        answer = _clean_text(source_row.get("answer"))
        image = source_row.get("image_path") or {}
        image_bytes = image.get("bytes") if isinstance(image, dict) else None
        if not question or not answer or not isinstance(image_bytes, (bytes, bytearray)):
            continue

        source_id = _clean_text(source_row.get("ID"))
        content = "<image>\n" + question
        if option:
            content += "\n\nOptions:\n" + option
        rows.append(
            {
                "prompt": [{"role": "user", "content": content}],
                "images": [{"bytes": bytes(image_bytes)}],
                "reward_model": {"ground_truth": answer},
                "data_source": "wemath_validation",
                "extra_info": {
                    "index": len(rows),
                    "source_id": source_id,
                    "split": _clean_text(source_row.get("split")),
                    "source_image": _clean_text(image.get("path"))
                    if isinstance(image, dict)
                    else "",
                    "validation_only": True,
                },
            }
        )
        preview.append(
            {
                "index": len(preview),
                "source_id": source_id,
                "prompt": content,
                "image_path": _clean_text(image.get("path"))
                if isinstance(image, dict)
                else "",
                "ground_truth": answer,
            }
        )
        if len(rows) == args.count:
            break

    if len(rows) != args.count:
        raise RuntimeError(f"Only found {len(rows)} valid multimodal rows; required {args.count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")
    args.preview.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in preview) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "preview": str(args.preview),
                "rows": len(rows),
                "schema": str(pa.Table.from_pylist(rows).schema),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
