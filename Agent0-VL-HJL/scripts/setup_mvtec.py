"""Setup MVTec AD dataset for Agent0-VL and HJL visual anomaly inspection.

Reuses and structures the local dataset from /mnt/d/Triad (default) into canonical
MVTec format under data/mvtec/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TRIAD_ROOT = Path("/mnt/d/Triad/evaluation/mvtec")
DEFAULT_OUTPUT_DIR = Path("data/mvtec")


def setup_mvtec_from_triad(
    triad_root: Path = DEFAULT_TRIAD_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    copy_files: bool = False,
) -> dict[str, Any]:
    """Import and organize MVTec dataset from Triad into canonical MVTec benchmark structure."""
    imgs_dir = triad_root / "imgs"
    musc_dir = triad_root / "musc"
    jsonl_path = triad_root / "question_musc.jsonl"

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Triad MVTec annotations not found at: {jsonl_path}")
    if not imgs_dir.is_dir():
        raise FileNotFoundError(f"Triad MVTec images directory not found at: {imgs_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Setting up MVTec dataset from {triad_root} into {output_dir}...")

    with jsonl_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(records)} sample records from {jsonl_path}")

    index_entries: list[dict[str, Any]] = []
    category_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"train_good": 0, "test_normal": 0, "test_defect": 0, "defects": set()}
    )

    # First pass: identify normal references for train/good per category
    category_normals: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for rec in records:
        if rec.get("gt") == 0:
            cat = rec["origin_path"].split("/")[0]
            src_img = imgs_dir / rec["image"]
            if src_img.is_file():
                category_normals[cat].append((rec, src_img))

    # Populate train/good reference corpus for HJL tool retrieval
    for cat, normals in category_normals.items():
        train_good_dir = output_dir / cat / "train" / "good"
        train_good_dir.mkdir(parents=True, exist_ok=True)
        for i, (rec, src_img) in enumerate(normals):
            fname = rec["origin_path"].split("/")[-1]
            dest_img = train_good_dir / fname
            if not dest_img.exists():
                if copy_files:
                    shutil.copy2(src_img, dest_img)
                else:
                    os.symlink(src_img.resolve(), dest_img)
            category_stats[cat]["train_good"] += 1

    # Second pass: populate test/ and ground_truth/ masks, and build index.jsonl
    for rec in records:
        origin_path = rec["origin_path"]
        parts = origin_path.split("/")
        cat = parts[0]
        split = parts[1]  # 'test'
        defect_type = parts[2]
        fname = parts[3]

        src_img = imgs_dir / rec["image"]
        if not src_img.is_file():
            logger.warning(f"Image file missing: {src_img}")
            continue

        test_dest_dir = output_dir / cat / "test" / defect_type
        test_dest_dir.mkdir(parents=True, exist_ok=True)
        dest_img = test_dest_dir / fname
        if not dest_img.exists():
            if copy_files:
                shutil.copy2(src_img, dest_img)
            else:
                os.symlink(src_img.resolve(), dest_img)

        # Mask if available
        mask_rel = rec.get("mask")
        dest_mask: Path | None = None
        if mask_rel:
            mask_fname = Path(mask_rel).name
            src_mask = musc_dir / mask_fname
            if src_mask.is_file():
                gt_dest_dir = output_dir / cat / "ground_truth" / defect_type
                gt_dest_dir.mkdir(parents=True, exist_ok=True)
                dest_mask = gt_dest_dir / fname
                if not dest_mask.exists():
                    if copy_files:
                        shutil.copy2(src_mask, dest_mask)
                    else:
                        os.symlink(src_mask.resolve(), dest_mask)

        is_anomaly = bool(rec.get("gt") == 1)
        if is_anomaly:
            category_stats[cat]["test_defect"] += 1
            category_stats[cat]["defects"].add(defect_type)
        else:
            category_stats[cat]["test_normal"] += 1

        entry = {
            "sample_id": f"mvtec_{cat}_{defect_type}_{Path(fname).stem}",
            "question_id": rec.get("question_id"),
            "category": cat,
            "split": split,
            "defect_type": defect_type,
            "is_anomaly": is_anomaly,
            "gt": rec.get("gt"),
            "image_path": str(dest_img),
            "mask_path": str(dest_mask) if dest_mask else None,
            "musc_score": rec.get("musc_scores"),
            "origin_path": origin_path,
        }
        index_entries.append(entry)

    # Write unified index.jsonl
    index_file = output_dir / "index.jsonl"
    with index_file.open("w", encoding="utf-8") as f:
        for entry in index_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Serialize manifest summary
    manifest = {
        "dataset": "mvtec_ad",
        "total_samples": len(index_entries),
        "total_categories": len(category_stats),
        "categories": {
            cat: {
                "train_good_references": stats["train_good"],
                "test_normal": stats["test_normal"],
                "test_defect": stats["test_defect"],
                "total_test": stats["test_normal"] + stats["test_defect"],
                "defect_types": sorted(stats["defects"]),
            }
            for cat, stats in sorted(category_stats.items())
        },
    }
    manifest_file = output_dir / "manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully organized MVTec dataset into {output_dir}")
    logger.info(f"Index written to {index_file} ({len(index_entries)} records)")
    logger.info(f"Manifest written to {manifest_file} ({len(category_stats)} categories)")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triad-root",
        type=Path,
        default=DEFAULT_TRIAD_ROOT,
        help="Path to Triad MVTec evaluation directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Target output directory for canonical MVTec dataset.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image files instead of creating symbolic links.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        manifest = setup_mvtec_from_triad(
            triad_root=args.triad_root,
            output_dir=args.output_dir,
            copy_files=args.copy,
        )
        print(f"\n=== MVTec AD Dataset Setup Complete ===")
        print(f"Total Categories: {manifest['total_categories']}")
        print(f"Total Test Samples: {manifest['total_samples']}")
        for cat, details in manifest["categories"].items():
            print(f"  {cat:15s}: {details['train_good_references']:2d} refs, {details['test_normal']:2d} normal, {details['test_defect']:3d} defect | {details['defect_types']}")
        return 0
    except Exception as exc:
        logger.error(f"Failed to setup dataset: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
