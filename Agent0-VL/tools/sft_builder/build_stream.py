"""Resumable, batch-wise construction of a complete SFT source split."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .build import BuildStats, build_records
from .dedup import record_hash
from .prompts import load_stage_solver_prompt
from .sources import SOURCE_STAGES, SourceFormatError, iter_source_samples
from .teacher import OpenAICompatibleTeacher, TeacherError


def _add_stats(total: BuildStats, part: BuildStats) -> None:
    for field in (
        "attempted",
        "exported",
        "solver_failures",
        "verifier_failures",
        "reference_failures",
        "unverified_skips",
        "duplicate_skips",
        "quality_failures",
    ):
        setattr(total, field, getattr(total, field) + getattr(part, field))
    for reason, count in part.failure_reasons.items():
        total.failure_reasons[reason] = total.failure_reasons.get(reason, 0) + count


def _state_path(output: Path) -> Path:
    return Path(str(output) + ".state.json")


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_existing_hashes(output: Path) -> set[str]:
    if not output.is_file():
        return set()
    hashes: set[str] = set()
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                hashes.add(record_hash(json.loads(line)))
    return hashes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_STAGES))
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--source-split", choices=("train", "dev", "test", "all"), default="train")
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--quality-profile", required=True, choices=("stage1", "stage2"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tasks", required=True, type=int)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-reasoning-steps", type=int, default=8)
    parser.add_argument("--sandbox-timeout", type=float, default=20.0)
    parser.add_argument("--teacher-base-url", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-top-p", type=float, default=1.0)
    parser.add_argument("--teacher-max-tokens", type=int, default=2048)
    parser.add_argument("--teacher-timeout", type=float, default=120.0)
    parser.add_argument("--teacher-retries", type=int, default=2)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_tasks <= 0 or args.concurrency <= 0 or args.batch_size <= 0:
        raise SystemExit("--max-tasks, --concurrency, and --batch-size must be positive")
    if args.max_reasoning_steps <= 0 or args.sandbox_timeout <= 0 or args.teacher_timeout <= 0 or args.teacher_retries < 0:
        raise SystemExit("timeouts and --max-reasoning-steps must be positive")

    output = args.output.expanduser()
    state_path = _state_path(output)
    processed = 0
    total = BuildStats()
    if args.resume:
        if not state_path.is_file():
            raise SystemExit(f"Cannot resume without state file: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        processed = int(state.get("processed_samples", 0))
        previous_stats = state.get("stats", {})
        for field in (
            "attempted",
            "exported",
            "solver_failures",
            "verifier_failures",
            "reference_failures",
            "unverified_skips",
            "duplicate_skips",
            "quality_failures",
        ):
            setattr(total, field, int(previous_stats.get(field, 0)))
        total.failure_reasons.update(previous_stats.get("failure_reasons", {}))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    seen = _read_existing_hashes(output)
    source_iter = iter_source_samples(
        args.source,
        args.source_path,
        stage=args.stage,
        source_split=args.source_split,
    )
    skipped = sum(1 for _ in itertools.islice(source_iter, processed))
    if skipped != processed:
        raise SystemExit(f"State says {processed} samples were processed, but source ended at {skipped}")

    teacher = OpenAICompatibleTeacher(
        args.teacher_base_url,
        args.teacher_model,
        api_key_env=args.teacher_api_key_env,
        temperature=args.teacher_temperature,
        top_p=args.teacher_top_p,
        max_tokens=args.teacher_max_tokens,
        timeout=args.teacher_timeout,
        retries=args.teacher_retries,
    )
    solver_prompt = load_stage_solver_prompt(args.repo_root, args.stage)
    remaining = args.max_tasks - processed
    with output.open("a", encoding="utf-8") as handle:
        while remaining > 0:
            batch = list(itertools.islice(source_iter, min(args.batch_size, remaining)))
            if not batch:
                break
            records, batch_stats = build_records(
                batch,
                teacher,
                solver_prompt,
                max_tasks=len(batch),
                max_reasoning_steps=args.max_reasoning_steps,
                sandbox_timeout=args.sandbox_timeout,
                concurrency=args.concurrency,
                quality_profile=args.quality_profile,
                batch_size=min(args.batch_size, len(batch)),
            )
            _add_stats(total, batch_stats)
            new_rows = 0
            for record in records:
                digest = record_hash(record)
                if digest in seen:
                    total.duplicate_skips += 1
                    continue
                seen.add(digest)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_rows += 1
            handle.flush()
            processed += len(batch)
            remaining -= len(batch)
            _write_state(state_path, {
                "source": args.source,
                "source_path": str(args.source_path),
                "source_split": args.source_split,
                "stage": args.stage,
                "quality_profile": args.quality_profile,
                "max_tasks": args.max_tasks,
                "processed_samples": processed,
                "exported_rows": len(seen),
                "last_batch_rows": new_rows,
                "stats": asdict(total),
                "complete": False,
            })
            print(json.dumps({
                "processed_samples": processed,
                "remaining": remaining,
                "new_rows": new_rows,
                "stats": asdict(total),
            }, ensure_ascii=False), flush=True)

    complete = remaining <= 0
    _write_state(state_path, {
        "source": args.source,
        "source_path": str(args.source_path),
        "source_split": args.source_split,
        "stage": args.stage,
        "quality_profile": args.quality_profile,
        "max_tasks": args.max_tasks,
        "processed_samples": processed,
        "exported_rows": len(seen),
        "stats": asdict(total),
        "complete": complete,
    })
    print(json.dumps({
        "output": str(output),
        "state": str(state_path),
        "complete": complete,
        "processed_samples": processed,
        "exported_rows": len(seen),
        "stats": asdict(total),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TeacherError, SourceFormatError) as exc:
        raise SystemExit(f"Build error: {exc}") from exc
