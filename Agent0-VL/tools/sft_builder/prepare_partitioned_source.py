"""Prepare a small, deterministic source subset for teacher-generated SFT.

The global partition files identify which raw rows belong to each SFT stage.
This utility materializes only those rows and rewrites their image references
to absolute paths inside the current workspace, so ``build_stream`` cannot
accidentally consume another stage or an RL row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def _rank(seed: int, row: dict[str, Any]) -> str:
    stable_id = str(row.get("source_record_id") or row.get("original_id") or row)
    return hashlib.sha256(f"{seed}:{stable_id}".encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve(root: Path, base: Path, value: Any, *, extra_dir: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid image reference: {value!r}")
    path = Path(value.strip())
    candidates: list[Path] = []
    if path.is_absolute():
        try:
            # Keep already-localized paths inside the current workspace.
            path.relative_to(root)
            candidates.append(path)
        except ValueError:
            pass
        # Some copied MM-Eureka rows retain the source workspace prefix.  Map
        # that prefix into the current project rather than depending on the
        # old workspace remaining mounted.
        old_project = Path("/mnt/d/Agent0/Agent0-VL")
        try:
            candidates.append(root / path.relative_to(old_project))
        except ValueError:
            pass
    else:
        candidates.append(base / path)
    if extra_dir:
        candidates.append(base / extra_dir / (path.name if path.is_absolute() else path))
    candidates.append(root / (path.name if path.is_absolute() else path))
    old_project = Path("/mnt/d/Agent0/Agent0-VL")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        # ``MMPR`` and ``K12`` may be symlinks left by the workspace copy.
        # Resolve them, then remap the old target into the current workspace
        # when the copied real files are present there.
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            resolved = candidate.absolute()
        try:
            relative = resolved.relative_to(old_project)
        except ValueError:
            relative = None
        if relative is not None:
            local_target = root / relative
            if local_target.is_file():
                return str(local_target.absolute())
        try:
            candidate.relative_to(root)
            return str(candidate.absolute())
        except ValueError:
            continue
    raise FileNotFoundError(f"Could not resolve image {value!r}; tried {candidates}")


def _load_raw(raw_path: Path, indices: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if raw_path.suffix.lower() == ".json":
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list: {raw_path}")
        for index in indices:
            rows[index] = data[index]
        return rows
    with raw_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in indices:
                rows[index] = json.loads(line)
            if len(rows) == len(indices):
                break
    return rows


def build(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    partition_path = (root / args.partition).resolve()
    raw_path = (root / args.raw).resolve()
    output = (root / args.output).resolve()
    metadata = [
        row
        for row in _read_jsonl(partition_path)
        if row.get("dataset") == args.dataset
        and row.get("usage_partition") == args.usage_partition
        and str(row.get("split")) == "train"
        and str(row.get("official_split")) in {"train", "train_inferred"}
    ]

    excluded_ids: set[str] = set()
    for manifest_path in args.exclude_manifest:
        manifest = json.loads(manifest_path.expanduser().read_text(encoding="utf-8"))
        excluded_ids.update(
            str(record_id)
            for record_id in manifest.get("source_record_ids", [])
            if record_id is not None
        )
    if excluded_ids:
        metadata = [
            row for row in metadata
            if str(row.get("source_record_id")) not in excluded_ids
        ]

    if args.stratify_image_subdataset:
        if args.dataset != "mulberry":
            raise ValueError("--stratify-image-subdataset is currently supported only for Mulberry")
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in metadata:
            image_ref = str(row.get("image_ref") or "")
            group = image_ref.split("/", 1)[0] or "(unknown)"
            groups.setdefault(group, []).append(row)
        for group_rows in groups.values():
            group_rows.sort(key=lambda row: (_rank(args.seed, row), str(row)))

        # Round-robin over the image-source folders so the bounded candidate
        # set represents small Mulberry subdatasets instead of mirroring only
        # the largest source collections.
        selected = []
        group_names = sorted(groups)
        offsets = {group: 0 for group in group_names}
        while len(selected) < args.count:
            made_progress = False
            for group in group_names:
                offset = offsets[group]
                if offset < len(groups[group]):
                    selected.append(groups[group][offset])
                    offsets[group] += 1
                    made_progress = True
                    if len(selected) == args.count:
                        break
            if not made_progress:
                break
        if len(selected) < args.count:
            raise ValueError(f"Only {len(selected)} eligible rows after exclusions; need {args.count}")
    else:
        metadata.sort(key=lambda row: (_rank(args.seed, row), str(row)))
        if len(metadata) < args.count:
            raise ValueError(f"Only {len(metadata)} eligible rows after exclusions; need {args.count}")
        selected = metadata[: args.count]
    raw_rows = _load_raw(raw_path, {int(row["row_index"]) for row in selected})
    raw_base = raw_path.parent
    output_rows: list[dict[str, Any]] = []
    for metadata_row in selected:
        index = int(metadata_row["row_index"])
        row = copy.deepcopy(raw_rows[index])
        if args.dataset == "mulberry":
            image_value = row.get("images")
            if isinstance(image_value, list):
                row["images"] = [
                    _resolve(root, raw_base, value, extra_dir="mulberry_images")
                    for value in image_value
                ]
            else:
                row["images"] = _resolve(root, raw_base, image_value, extra_dir="mulberry_images")
        elif args.dataset == "mm_eureka":
            image_values = row.get("image_urls") or row.get("images") or row.get("image")
            if not isinstance(image_values, list):
                image_values = [image_values]
            row["image_urls"] = [
                _resolve(root, raw_base, value)
                for value in image_values
            ]
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        output_rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "partition": str(partition_path),
                "raw": str(raw_path),
                "output": str(output),
                "usage_partition": args.usage_partition,
                "split_policy": "train and train_inferred only",
                "rows": len(output_rows),
                "seed": args.seed,
                "excluded_manifests": [str(path) for path in args.exclude_manifest],
                "stratify_by": "image_ref first path segment" if args.stratify_image_subdataset else None,
                "source_record_ids": [row.get("source_record_id") for row in selected],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "rows": len(output_rows), "manifest": str(manifest)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=("mulberry", "mm_eureka"), required=True)
    parser.add_argument("--usage-partition", required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Exclude source_record_ids recorded in a previous candidate manifest; may be repeated.",
    )
    parser.add_argument(
        "--stratify-image-subdataset",
        action="store_true",
        help="For Mulberry, round-robin candidate selection across image_ref subdataset folders.",
    )
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
