"""Evaluate benchmarks through a probed Responses endpoint."""

from __future__ import annotations

import argparse
import sys

from verl.evaluation.agent0_evaluator import Agent0Evaluator, EvaluatorConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", required=True, help="Comma-separated benchmark names")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_per_sample_results", type=str, default="true")
    parser.add_argument("--save_intermediate", type=str, default="true")
    parser.add_argument("--intermediate_save_interval", type=int, default=100)
    args = parser.parse_args()
    evaluator = Agent0Evaluator(EvaluatorConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_per_sample_results=args.save_per_sample_results.lower() == "true",
        save_intermediate=args.save_intermediate.lower() == "true",
        intermediate_save_interval=args.intermediate_save_interval,
    ))
    evaluator.evaluate_benchmarks(
        [name.strip() for name in args.benchmarks.split(",") if name.strip()],
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
