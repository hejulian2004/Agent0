"""Build the bounded RL task-row smoke dataset.

The input is the completed local-Qwen SFT smoke selection.  This command does
not call a model: it converts the same selected source tasks into the RLHFDataset
Parquet contract, with one task row per sample and binary image payloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.data_builder.exporters.rl_verl import (
    RL_DATASETS,
    RLSmokeExportError,
    build_rl_smoke_rows,
    write_rl_smoke_parquet,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _repo_root()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "smoke" / "local_qwen_27b_10_per_dataset" / "sft_records.jsonl",
        help="completed SFT smoke JSONL used as the deterministic source selection",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "smoke" / "local_qwen_27b_10_per_dataset" / "rl",
    )
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument(
        "--allow-eval-only",
        action="store_true",
        help="include eval/test rows for explicitly debug-only smoke output",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    if not input_path.is_file():
        raise SystemExit(f"input JSONL does not exist: {input_path}")
    try:
        rows, report = build_rl_smoke_rows(
            input_path,
            samples_per_dataset=args.samples_per_dataset,
            datasets=RL_DATASETS,
            allow_eval_only=args.allow_eval_only,
        )
        outputs = write_rl_smoke_parquet(rows, output_dir)
    except RLSmokeExportError as exc:
        raise SystemExit(str(exc)) from exc

    report["input"] = str(input_path)
    report["output_dir"] = str(output_dir)
    report["outputs"] = outputs
    report_path = output_dir / "rl_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selection_path = output_dir / "rl_selection_manifest.jsonl"
    with selection_path.open("w", encoding="utf-8") as handle:
        for item in report["source_selection"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output_dir": str(output_dir),
        "rl_task_row_count": report["rl_task_row_count"],
        "dataset_counts": report["dataset_counts"],
        "outputs": outputs,
        "formal_training_eligible_row_count": report["formal_training_eligible_row_count"],
        "eval_only_debug_row_count": report["eval_only_debug_row_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
