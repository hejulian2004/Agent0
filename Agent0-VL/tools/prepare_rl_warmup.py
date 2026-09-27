"""Build a disjoint, deterministic external-reward warm-up RL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _key(row: dict) -> str:
    payload = {
        "prompt": row["prompt"],
        "ground_truth": row["reward_model"]["ground_truth"],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build(source: Path, formal: Path, output: Path, preview: Path, size: int, seed: int) -> None:
    formal_rows = pq.read_table(formal).to_pylist()
    source_rows = pq.read_table(source).to_pylist()
    formal_keys = {_key(row) for row in formal_rows}

    candidates = [row for row in source_rows if _key(row) not in formal_keys]
    if len(candidates) < size:
        raise ValueError(f"only {len(candidates)} disjoint rows available; need {size}")

    # Round-robin by source keeps the small warm-up set from being dominated by
    # one source while remaining completely reproducible.
    groups: dict[str, deque[dict]] = defaultdict(deque)
    for row in sorted(candidates, key=lambda r: (_key(r), str(r["extra_info"].get("source_id", "")))):
        groups[str(row["extra_info"].get("source_id", "unknown"))].append(row)
    source_ids = sorted(groups)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < size:
        source_id = source_ids[cursor % len(source_ids)]
        if groups[source_id]:
            selected.append(groups[source_id].popleft())
        cursor += 1
        if cursor >= len(source_ids) and not any(groups.values()):
            break
    if len(selected) != size:
        raise AssertionError(f"selected {len(selected)} rows, expected {size}")

    table = pa.Table.from_pylist(selected, schema=pq.read_table(source).schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output, compression="zstd")

    preview.parent.mkdir(parents=True, exist_ok=True)
    with preview.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            prompt = row["prompt"]
            text = "\n".join(str(message.get("content", "")) for message in prompt)
            handle.write(json.dumps({
                "index": index,
                "source_id": row["extra_info"].get("source_id"),
                "prompt": text,
                "ground_truth": row["reward_model"]["ground_truth"],
                "image_count": len(row.get("images", [])),
            }, ensure_ascii=False) + "\n")

    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "source": str(source),
        "formal_dataset": str(formal),
        "output": str(output),
        "rows": len(selected),
        "disjoint_by_prompt_and_ground_truth": True,
        "selection": "deterministic source round-robin",
        "seed": seed,
        "data_sources": sorted({str(row["data_source"]) for row in selected}),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(output),
        "rows": len(selected),
        "available_disjoint_rows": len(candidates),
        "unique_source_ids": len({str(row["extra_info"].get("source_id", "unknown")) for row in selected}),
        "preview": str(preview),
        "manifest": str(manifest),
    }, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/rl/rl_1000.parquet"))
    parser.add_argument("--formal", type=Path, default=Path("data/rl/rl_200.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/rl/rl_warmup_200.parquet"))
    parser.add_argument("--preview", type=Path, default=Path("data/rl/rl_warmup_200.preview.jsonl"))
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    build(args.source, args.formal, args.output, args.preview, args.size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
