"""Batch evaluation script for MVTec AD visual anomaly detection.

Supports evaluating direct, react, react_verifier, and full HJL modes
across data/mvtec/index.jsonl, computing:
1. Image-level classification metrics: Accuracy, Precision, Recall, Specificity (TNR), F1
2. Operational efficiency: Fast-path rate, Average Steps, Tool Cost, Latency
3. Defect localization hit rate (IoU > 0.5 against ground-truth mask bounding box)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from hjl.config import HJLConfig
from hjl.engine import HJLEngine

logger = logging.getLogger(__name__)


def compute_mask_bbox(mask_path: str | Path | None) -> list[int] | None:
    """Extract minimum bounding box [x1, y1, x2, y2] from ground-truth binary mask."""
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.is_file():
        return None
    try:
        with Image.open(path) as mask:
            arr = np.array(mask.convert("L"))
        ys, xs = np.where(arr > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    except Exception:
        return None


def compute_iou(box_a: list[int] | None, box_b: list[int] | None) -> float:
    """Compute Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def run_batch_evaluation(
    index_path: Path = Path("data/mvtec/index.jsonl"),
    output_dir: Path = Path("outputs/eval_results"),
    mode: str = "hjl",
    categories: list[str] | None = None,
    limit: int | None = None,
    max_steps: int | None = None,
    mock: bool = False,
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    """Run batch evaluation over MVTec index records and aggregate metrics."""
    if not index_path.is_file():
        raise FileNotFoundError(f"MVTec index file not found at {index_path}. Run scripts/setup_mvtec.py first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    details_file = output_dir / f"eval_{mode}_{timestamp}_details.jsonl"
    summary_file = output_dir / f"eval_{mode}_{timestamp}_summary.json"

    # Load records from index
    records: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter categories if requested
    if categories:
        target_cats = set(categories)
        records = [r for r in records if r.get("category") in target_cats]

    if limit is not None and limit > 0:
        records = records[:limit]

    logger.info(f"Loaded {len(records)} samples for evaluation in mode: {mode} (mock={mock})")

    # Initialize Engine
    cfg = HJLConfig.from_yaml(config_path)
    if max_steps is not None:
        cfg.max_steps = max_steps
    engine = HJLEngine(config=cfg, mock=mock)

    results: list[dict[str, Any]] = []

    tp, fp, tn, fn = 0, 0, 0, 0
    fast_path_count = 0
    localization_hits = 0
    localization_candidates = 0
    total_steps = 0
    total_tool_cost = 0
    total_latency = 0.0

    category_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0})

    for idx, rec in enumerate(records):
        sample_id = rec["sample_id"]
        category = rec["category"]
        image_path = rec["image_path"]
        gt_is_anomaly = bool(rec["gt"] == 1)
        mask_path = rec.get("mask_path")

        logger.info(f"[{idx+1}/{len(records)}] Inspecting {sample_id} ({category}) | GT Anomaly: {gt_is_anomaly}")
        start_t = time.time()

        pred_anomaly: bool | None = None
        detected_regions: list[list[int]] = []
        steps = 0
        tool_cost = 0
        stop_reason = ""
        error: str | None = None

        try:
            if mode == "direct":
                res = engine.run_direct(image_path=image_path, category=category)
                pred_anomaly = res["is_anomaly"]
                steps = res["total_steps"]
                tool_cost = res["tool_cost"]
                stop_reason = res["stop_reason"]
            elif mode == "react":
                res = engine.run_react(image_path=image_path, category=category, max_steps=cfg.max_steps)
                pred_anomaly = res["is_anomaly"]
                steps = res["total_steps"]
                tool_cost = res["tool_cost"]
                stop_reason = res["stop_reason"]
            elif mode == "react_verifier":
                res = engine.run_react_verifier(image_path=image_path, category=category, max_steps=cfg.max_steps)
                pred_anomaly = res["is_anomaly"]
                steps = res["total_steps"]
                tool_cost = res["tool_cost"]
                stop_reason = res["stop_reason"]
            elif mode == "hjl":
                state = engine.run_hjl(image_path=image_path, category=category, max_steps=cfg.max_steps)
                final_p = state.final_prediction or {}
                pred_anomaly = final_p.get("is_anomaly")
                if pred_anomaly is None and final_p.get("best_effort"):
                    pred_anomaly = final_p.get("best_effort_is_anomaly")
                detected_regions = final_p.get("detected_regions", [])
                steps = state.current_step
                tool_cost = state.evidence_state.tool_cost
                stop_reason = str(state.stop_reason.value if state.stop_reason else "UNKNOWN")
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")

        except Exception as exc:
            logger.error(f"Execution error on sample {sample_id}: {exc}", exc_info=True)
            error = str(exc)
            pred_anomaly = False
            stop_reason = "ERROR"

        latency = time.time() - start_t
        total_latency += latency
        total_steps += steps
        total_tool_cost += tool_cost

        # Fast path check (0 tool cost / 0 step confirmation)
        if tool_cost == 0 and stop_reason == "CONFIRMED_NORMAL":
            fast_path_count += 1

        # Binary confusion matrix
        predicted_as_anomaly = bool(pred_anomaly is True)
        if gt_is_anomaly and predicted_as_anomaly:
            tp += 1
            category_stats[category]["tp"] += 1
        elif not gt_is_anomaly and predicted_as_anomaly:
            fp += 1
            category_stats[category]["fp"] += 1
        elif not gt_is_anomaly and not predicted_as_anomaly:
            tn += 1
            category_stats[category]["tn"] += 1
        else:
            fn += 1
            category_stats[category]["fn"] += 1
        category_stats[category]["total"] += 1

        # Localization hit check for defective samples with masks
        loc_hit = False
        gt_bbox = None
        if gt_is_anomaly and mask_path:
            gt_bbox = compute_mask_bbox(mask_path)
            if gt_bbox:
                localization_candidates += 1
                for d_box in detected_regions:
                    if compute_iou(d_box, gt_bbox) >= 0.5:
                        loc_hit = True
                        break
                if loc_hit:
                    localization_hits += 1

        record_result = {
            "sample_id": sample_id,
            "category": category,
            "defect_type": rec.get("defect_type"),
            "gt_anomaly": gt_is_anomaly,
            "pred_anomaly": pred_anomaly,
            "is_correct": bool(gt_is_anomaly == predicted_as_anomaly),
            "stop_reason": stop_reason,
            "steps": steps,
            "tool_cost": tool_cost,
            "latency_seconds": round(latency, 3),
            "gt_bbox": gt_bbox,
            "detected_regions": detected_regions,
            "localization_hit": loc_hit if gt_is_anomaly and gt_bbox else None,
            "error": error,
        }
        results.append(record_result)

        # Write streaming jsonl
        with details_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record_result, ensure_ascii=False) + "\n")

    # Aggregate global metrics
    total_samples = len(records)
    accuracy = (tp + tn) / max(1, total_samples)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    loc_acc = localization_hits / max(1, localization_candidates)
    fast_path_rate = fast_path_count / max(1, total_samples)
    avg_steps = total_steps / max(1, total_samples)
    avg_tool_cost = total_tool_cost / max(1, total_samples)
    avg_latency = total_latency / max(1, total_samples)

    summary = {
        "mode": mode,
        "mock": mock,
        "total_samples": total_samples,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall_tpr": round(recall, 4),
        "specificity_tnr": round(specificity, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "localization": {
            "evaluable_samples": localization_candidates,
            "hits_iou_ge_05": localization_hits,
            "hit_rate": round(loc_acc, 4),
        },
        "efficiency": {
            "fast_path_rate": round(fast_path_rate, 4),
            "fast_path_count": fast_path_count,
            "avg_steps": round(avg_steps, 2),
            "avg_tool_cost": round(avg_tool_cost, 2),
            "avg_latency_seconds": round(avg_latency, 3),
            "total_latency_seconds": round(total_latency, 2),
        },
        "per_category": {
            cat: {
                "total": stats["total"],
                "acc": round((stats["tp"] + stats["tn"]) / max(1, stats["total"]), 4),
                "tp": stats["tp"],
                "fp": stats["fp"],
                "tn": stats["tn"],
                "fn": stats["fn"],
            }
            for cat, stats in sorted(category_stats.items())
        },
    }

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"         MVTec AD Benchmark Results ({mode.upper()})")
    print("=" * 70)
    print(f"Total Samples: {total_samples:<6} | Mode: {mode:<12} | Mock: {mock}")
    print("-" * 70)
    print(f"Accuracy:    {accuracy * 100:.2f}%")
    print(f"Precision:   {precision * 100:.2f}%")
    print(f"Recall (TPR):{recall * 100:.2f}% (Defect identification)")
    print(f"Specificity: {specificity * 100:.2f}% (Normal pass rate)")
    print(f"F1 Score:    {f1:.4f}")
    print("-" * 70)
    print(f"Fast-path Exit Rate: {fast_path_rate * 100:.2f}% ({fast_path_count}/{total_samples} samples)")
    print(f"Avg Steps:           {avg_steps:.2f}")
    print(f"Avg Tool Cost:       {avg_tool_cost:.2f}")
    print(f"Avg Latency:         {avg_latency:.2f} s/sample")
    if localization_candidates > 0:
        print(f"BBox Hit Rate:       {loc_acc * 100:.2f}% (IoU >= 0.5 on {localization_candidates} defect samples)")
    print("-" * 70)
    print(f"Details saved: {details_file}")
    print(f"Summary saved: {summary_file}")
    print("=" * 70 + "\n")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/mvtec/index.jsonl"), help="MVTec index file path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_results"), help="Results output dir.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "react", "react_verifier", "hjl"],
        default="hjl",
        help="Evaluation mode.",
    )
    parser.add_argument("--category", type=str, nargs="+", default=None, help="Filter categories (e.g. bottle cable).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate.")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per sample.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock model caller.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        run_batch_evaluation(
            index_path=args.index,
            output_dir=args.output_dir,
            mode=args.mode,
            categories=args.category,
            limit=args.limit,
            max_steps=args.max_steps,
            mock=args.mock,
            config_path=args.config,
        )
        return 0
    except Exception as exc:
        logger.error(f"Batch evaluation failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
