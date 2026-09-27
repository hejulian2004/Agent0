"""Select reproducible broad-frame SFT and RL subsets.

The existing small training files were selected from the beginning of their
candidate files.  This utility keeps the same row counts and schemas, but
selects by a seeded hash rank over the complete available candidate frame.
Stage-1 images are copied into the workspace so the resulting file is
self-contained and does not depend on the old source workspace path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from tools.sft_builder.dedup import record_hash
from tools.sft_builder.merge_sft import audit_record


DEFAULT_SEED = 20260922
OLD_WORKSPACE = Path("/mnt/d/Agent0/Agent0-VL")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _rank(seed: int, namespace: str, row: Mapping[str, Any]) -> str:
    digest = record_hash(dict(row))
    payload = f"{seed}:{namespace}:{digest}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        digest = record_hash(row)
        if digest in seen:
            continue
        seen.add(digest)
        result.append(row)
    return result


def _resolve_local_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    candidates: list[Path] = []
    try:
        candidates.append(project_root / path.relative_to(OLD_WORKSPACE))
    except ValueError:
        candidates.append(path)
    if value.startswith("/mnt/d/Agent0/Agent0-VL/"):
        candidates.append(project_root / value.split("/mnt/d/Agent0/Agent0-VL/", 1)[1])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve SFT image {value!r}")


def _localize_stage1(
    rows: list[dict[str, Any]],
    output: Path,
    images_dir: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    localized: list[dict[str, Any]] = []
    image_count = 0
    images_dir.mkdir(parents=True, exist_ok=True)
    for row_index, original in enumerate(rows, 1):
        row = copy.deepcopy(original)
        new_images: list[str] = []
        for image_index, value in enumerate(row.get("images", []), 1):
            if not isinstance(value, str) or value.startswith(("data:", "http://", "https://")):
                new_images.append(value)
                continue
            source = _resolve_local_path(value, project_root)
            suffix = source.suffix.lower() or ".bin"
            destination = images_dir / f"{row_index:04d}_{image_index:02d}{suffix}"
            shutil.copy2(source, destination)
            try:
                relative = destination.relative_to(project_root)
            except ValueError as exc:
                raise RuntimeError(f"Image output escaped project root: {destination}") from exc
            new_images.append(relative.as_posix())
            image_count += 1
        row["images"] = new_images
        localized.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in localized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return localized, image_count


def _write_sft(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    stage: int,
    count: int,
    seed: int,
    project_root: Path,
    images_dir: Path | None = None,
) -> dict[str, Any]:
    input_rows = _read_jsonl(input_path)
    unique_rows = _deduplicate(input_rows)
    if len(unique_rows) < count:
        raise ValueError(f"{input_path} has only {len(unique_rows)} unique rows; need {count}")

    ranked = sorted(
        enumerate(unique_rows),
        key=lambda item: (_rank(seed, f"sft-stage-{stage}", item[1]), item[0]),
    )
    selected_pairs = ranked[:count]
    selected = [row for _, row in selected_pairs]
    selected_input_indices = [index for index, _ in selected_pairs]
    selected_hashes = [record_hash(row) for row in selected]

    source_counts: Counter[str] = Counter()
    if stage == 1:
        for row in selected:
            images = row.get("images", [])
            source = str(images[0]) if images else "no-image"
            if "geometry3k" in source:
                source = "geometry3k"
            elif "geoqa" in source:
                source = "geoqa"
            else:
                source = "other"
            source_counts[source] += 1

    if stage == 1:
        if images_dir is None:
            raise ValueError("Stage 1 requires an image output directory")
        selected, images_localized = _localize_stage1(
            selected, output_path, images_dir, project_root
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        images_localized = 0

    rejected: Counter[str] = Counter()
    for row in selected:
        reason = audit_record(row, stage)
        if reason is not None:
            rejected[reason] += 1
    if rejected:
        raise ValueError(f"Selected rows failed strict Stage-{stage} audit: {dict(rejected)}")

    manifest = {
        "stage": stage,
        "input": str(input_path),
        "output": str(output_path),
        "candidate_rows": len(input_rows),
        "unique_candidate_rows": len(unique_rows),
        "selected_rows": len(selected),
        "strict_audit": True,
        "selection": "seeded sha256 rank over complete unique candidate pool",
        "seed": seed,
        "selected_input_indices": selected_input_indices,
        "selected_hashes": selected_hashes,
        "source_counts": dict(source_counts),
        "images_localized": images_localized,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _rl_key(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"prompt": row["prompt"], "ground_truth": row["reward_model"]["ground_truth"]},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_rl(
    *,
    input_path: Path,
    output_path: Path,
    preview_path: Path,
    manifest_path: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    table = pq.read_table(input_path)
    rows = table.to_pylist()
    if len(rows) < count:
        raise ValueError(f"{input_path} has only {len(rows)} rows; need {count}")

    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(_rl_key(row), row)
    ranked = sorted(
        unique.items(),
        key=lambda item: (hashlib.sha256(f"{seed}:rl:{item[0]}".encode()).hexdigest(), item[0]),
    )
    selected = [row for _, row in ranked[:count]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected, schema=table.schema), output_path, compression="zstd")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with preview_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            prompt = row["prompt"]
            text = "\n".join(str(message.get("content", "")) for message in prompt)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "source_id": row.get("extra_info", {}).get("source_id"),
                        "source_index": row.get("extra_info", {}).get("index"),
                        "data_source": row.get("data_source"),
                        "prompt": text,
                        "ground_truth": row["reward_model"]["ground_truth"],
                        "image_count": len(row.get("images", [])),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    source_indices = [int(row.get("extra_info", {}).get("index", -1)) for row in selected]
    manifest = {
        "source": str(input_path),
        "output": str(output_path),
        "preview": str(preview_path),
        "candidate_rows": len(rows),
        "unique_candidate_rows": len(unique),
        "selected_rows": len(selected),
        "selection": "seeded sha256 rank over complete unique candidate pool",
        "seed": seed,
        "source_index_range": [min(source_indices), max(source_indices)],
        "data_sources": sorted({str(row.get("data_source")) for row in selected}),
        "validation_only_counts": dict(Counter(bool(row.get("extra_info", {}).get("validation_only")) for row in selected)),
        "selected_keys": [_rl_key(row) for row in selected],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stage1-input", type=Path, default=Path("data/sft/large/stage1_pool.jsonl"))
    parser.add_argument("--stage1-output", type=Path, default=Path("data/sft/large/stage1_500_broad_local.jsonl"))
    parser.add_argument("--stage1-images", type=Path, default=Path("data/sft/images/stage1_500_broad"))
    parser.add_argument("--stage2-input", type=Path, default=Path("data/sft/large/stage2_pool.jsonl"))
    parser.add_argument("--stage2-output", type=Path, default=Path("data/sft/large/stage2_500_broad.jsonl"))
    parser.add_argument("--rl-input", type=Path, default=Path("data/rl/rl_1000.parquet"))
    parser.add_argument("--rl-output", type=Path, default=Path("data/rl/rl_200_broad.parquet"))
    parser.add_argument("--rl-preview", type=Path, default=Path("data/rl/rl_200_broad.preview.jsonl"))
    parser.add_argument("--rl-manifest", type=Path, default=Path("data/rl/rl_200_broad.manifest.json"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    paths = [
        args.stage1_input,
        args.stage1_output,
        args.stage1_images,
        args.stage2_input,
        args.stage2_output,
        args.rl_input,
        args.rl_output,
        args.rl_preview,
        args.rl_manifest,
    ]
    resolved = [path if path.is_absolute() else root / path for path in paths]
    stage1_input, stage1_output, stage1_images, stage2_input, stage2_output, rl_input, rl_output, rl_preview, rl_manifest = resolved

    stage1_manifest = _write_sft(
        input_path=stage1_input,
        output_path=stage1_output,
        manifest_path=stage1_output.with_suffix(".manifest.json"),
        stage=1,
        count=500,
        seed=args.seed,
        project_root=root,
        images_dir=stage1_images,
    )
    stage2_manifest = _write_sft(
        input_path=stage2_input,
        output_path=stage2_output,
        manifest_path=stage2_output.with_suffix(".manifest.json"),
        stage=2,
        count=500,
        seed=args.seed,
        project_root=root,
    )
    rl_manifest_data = _write_rl(
        input_path=rl_input,
        output_path=rl_output,
        preview_path=rl_preview,
        manifest_path=rl_manifest,
        count=200,
        seed=args.seed,
    )
    print(json.dumps({"stage1": stage1_manifest, "stage2": stage2_manifest, "rl": rl_manifest_data}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
