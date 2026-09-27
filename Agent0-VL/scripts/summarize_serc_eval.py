#!/usr/bin/env python3
"""Summarize per-trajectory SERC validation results for model comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--val-n", required=True, type=int)
    return parser.parse_args()


def _flatten_samples(raw: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for prompt_rows in raw.values():
        if not isinstance(prompt_rows, dict):
            continue
        for rows in prompt_rows.values():
            if isinstance(rows, dict):
                rows = [rows]
            if isinstance(rows, list):
                samples.extend(row for row in rows if isinstance(row, dict))
    return samples


def _sum(samples: list[dict[str, Any]], key: str) -> int:
    return sum(int(sample.get(key, 0) or 0) for sample in samples)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _benchmark_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(sample.get("is_correct", False)) for sample in samples)
    solver_steps = _sum(samples, "num_steps")
    verifier_triggers = _sum(samples, "verifier_trigger_count")
    verifier_calls = _sum(samples, "verifier_call_count")
    valid_verifier_calls = _sum(samples, "valid_verifier_calls")
    repair_triggers = _sum(samples, "repair_trigger_count")
    repair_applied = _sum(samples, "repair_applied_count")
    repair_successes = _sum(samples, "repair_success_count")
    tool_calls = _sum(samples, "tool_call_count")
    successful_tool_calls = _sum(samples, "successful_tool_calls")
    return {
        "trajectory_count": len(samples),
        "correct_trajectories": correct,
        "answer_accuracy": _rate(correct, len(samples)),
        "solver_steps": solver_steps,
        "tool_call_count": tool_calls,
        "successful_tool_calls": successful_tool_calls,
        "tool_call_success_rate": _rate(successful_tool_calls, tool_calls),
        "verifier_trigger_count": verifier_triggers,
        "verifier_trigger_rate": _rate(verifier_triggers, solver_steps),
        "verifier_call_count": verifier_calls,
        "valid_verifier_calls": valid_verifier_calls,
        "verifier_call_success_rate": _rate(valid_verifier_calls, verifier_calls),
        "repair_trigger_count": repair_triggers,
        "repair_trigger_rate_per_verifier_step": _rate(repair_triggers, verifier_triggers),
        "repair_trigger_rate_per_solver_step": _rate(repair_triggers, solver_steps),
        "repair_applied_count": repair_applied,
        "repair_application_rate": _rate(repair_applied, repair_triggers),
        "repair_success_count": repair_successes,
        "repair_success_rate": _rate(repair_successes, repair_applied),
        "repair_score_improvement_count": _sum(samples, "repair_score_improvement_count"),
        "repair_score_improvement_rate": _rate(
            _sum(samples, "repair_score_improvement_count"), repair_applied
        ),
    }


def main() -> int:
    args = _parse_args()
    raw = json.loads(args.results_json.read_text(encoding="utf-8"))
    samples = _flatten_samples(raw)
    if not samples:
        raise RuntimeError(f"No per-trajectory records found in {args.results_json}")

    prompt_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    benchmark_samples: dict[str, list[dict[str, Any]]] = {}
    for source_rows in raw.values():
        if not isinstance(source_rows, dict):
            continue
        for prompt, rows in source_rows.items():
            if isinstance(rows, dict):
                rows = [rows]
            if isinstance(rows, list):
                valid_rows = [row for row in rows if isinstance(row, dict)]
                source = str(valid_rows[0].get("data_source", "unknown")) if valid_rows else "unknown"
                prompt_groups[(source, str(prompt))] = valid_rows
                benchmark_samples.setdefault(source, []).extend(valid_rows)

    correct = sum(bool(sample.get("is_correct", False)) for sample in samples)
    solver_steps = _sum(samples, "num_steps")
    verifier_triggers = _sum(samples, "verifier_trigger_count")
    verifier_calls = _sum(samples, "verifier_call_count")
    valid_verifier_calls = _sum(samples, "valid_verifier_calls")
    repair_triggers = _sum(samples, "repair_trigger_count")
    repair_applied = _sum(samples, "repair_applied_count")
    repair_successes = _sum(samples, "repair_success_count")
    repair_improvements = _sum(samples, "repair_score_improvement_count")
    tool_calls = _sum(samples, "tool_call_count")
    successful_tool_calls = _sum(samples, "successful_tool_calls")

    prompt_passes = sum(
        any(bool(sample.get("is_correct", False)) for sample in rows)
        for rows in prompt_groups.values()
        if rows
    )
    prompts_with_results = sum(bool(rows) for rows in prompt_groups.values())

    summary = {
        "schema_version": "agent0vl.serc_eval_summary.v1",
        "model_path": args.model_path,
        "dataset": args.dataset,
        "physical_gpus": args.gpu_ids,
        "samples_per_prompt": args.val_n,
        "prompt_count": prompts_with_results,
        "trajectory_count": len(samples),
        "correct_trajectories": correct,
        "answer_accuracy": _rate(correct, len(samples)),
        "macro_average_benchmark_accuracy": (
            sum(float(metrics["answer_accuracy"]) for metrics in
                (_benchmark_summary(rows) for rows in benchmark_samples.values())
                if metrics["answer_accuracy"] is not None) / len(benchmark_samples)
            if benchmark_samples else None
        ),
        "per_benchmark": {
            benchmark: _benchmark_summary(rows)
            for benchmark, rows in sorted(benchmark_samples.items())
        },
        "prompts_with_any_correct_trajectory": prompt_passes,
        "pass_at_n_accuracy": _rate(prompt_passes, prompts_with_results),
        "solver_steps": solver_steps,
        "tool_call_count": tool_calls,
        "successful_tool_calls": successful_tool_calls,
        "tool_call_success_rate": _rate(successful_tool_calls, tool_calls),
        "verifier_trigger_count": verifier_triggers,
        "verifier_trigger_rate": _rate(verifier_triggers, solver_steps),
        "verifier_call_count": verifier_calls,
        "valid_verifier_calls": valid_verifier_calls,
        "verifier_call_success_rate": _rate(valid_verifier_calls, verifier_calls),
        "repair_trigger_count": repair_triggers,
        "repair_trigger_rate_per_verifier_step": _rate(repair_triggers, verifier_triggers),
        "repair_trigger_rate_per_solver_step": _rate(repair_triggers, solver_steps),
        "repair_applied_count": repair_applied,
        "repair_application_rate": _rate(repair_applied, repair_triggers),
        "repair_success_count": repair_successes,
        "repair_success_rate": _rate(repair_successes, repair_applied),
        "repair_success_definition": (
            "Among applied repairs, post-repair Verifier confidence is at least the 0.7 repair threshold."
        ),
        "repair_score_improvement_count": repair_improvements,
        "repair_score_improvement_rate": _rate(repair_improvements, repair_applied),
        "repair_score_improvement_definition": (
            "Among applied repairs, the post-repair Verifier score is greater than its pre-repair score."
        ),
        "per_trajectory_results": str(args.results_json),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
