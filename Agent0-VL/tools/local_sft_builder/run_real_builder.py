"""Phase 2A dry-run entry point.

This command performs source preflight and deterministic sampling only.  It
intentionally has no non-dry execution path yet, so a successful invocation
cannot consume a Teacher request slot or create generation artifacts.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .budget import TeacherRequestBudget
from .run_manifest import build_phase2a_manifest, write_manifest
from .source_adapter import (
    ForbiddenSourceIndex,
    RealSourceAdapter,
    SourcePreflightError,
)
from .source_guard import SourceLeakageGuard


EXPECTED_BASE_SHA = "f775b5101e62fe92976831adf4a21a38fcc0a767"


class RunDirectoryConflict(RuntimeError):
    """Raised when a run would overwrite an existing directory."""


def _repo_root() -> Path:
    # .../Agent0-dev/Agent0-VL/tools/local_sft_builder/run_real_builder.py
    return Path(__file__).resolve().parents[3]


def _git_revision(repo_root: Path, revision: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "rev-parse",
            "--verify",
            f"{revision}^{{commit}}",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot resolve git commit: {revision}")
    return completed.stdout.strip()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _frozen_base_sha(value: str) -> str:
    if value != EXPECTED_BASE_SHA:
        raise argparse.ArgumentTypeError(
            f"must equal the frozen Phase 2A base SHA {EXPECTED_BASE_SHA}"
        )
    return value


def _write_jsonl(path: Path, values: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _close_empty_ledger(db_path: Path) -> None:
    """Leave the preflight ledger as one portable SQLite file."""

    connection = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.commit()
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2A source preflight without Teacher generation."
    )
    parser.add_argument("--source-path", action="append", required=True)
    parser.add_argument("--forbidden-source-path", action="append", default=[])
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--image-root-config")
    parser.add_argument("--max-tasks", type=_positive_int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base-sha",
        type=_frozen_base_sha,
        required=True,
        help=f"frozen Phase 2A base commit ({EXPECTED_BASE_SHA})",
    )
    parser.add_argument("--phase1-freeze-sha", required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.dry_run:
        raise ValueError("Phase 2A currently supports --dry-run only")

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else _repo_root()
    output_dir = Path(args.output_run_dir).expanduser().resolve()
    if output_dir.exists():
        raise RunDirectoryConflict(f"refusing to overwrite existing run directory: {output_dir}")

    if args.base_sha != EXPECTED_BASE_SHA:
        raise ValueError(
            f"Phase 2A base SHA must equal the frozen commit {EXPECTED_BASE_SHA}"
        )
    base_sha = _git_revision(repo_root, args.base_sha)
    if base_sha != EXPECTED_BASE_SHA:
        raise ValueError(
            f"resolved Phase 2A base SHA does not equal the frozen commit {EXPECTED_BASE_SHA}"
        )
    phase1_freeze_sha = _git_revision(repo_root, args.phase1_freeze_sha)
    builder_commit_sha = _git_revision(repo_root, "HEAD")

    forbidden_index = ForbiddenSourceIndex.from_paths(args.forbidden_source_path)
    adapter = RealSourceAdapter(
        args.source_path,
        stage=args.stage,
        image_root=args.image_root,
        image_root_config=args.image_root_config,
        source_guard=SourceLeakageGuard(),
        forbidden_index=forbidden_index,
    )
    result = adapter.preflight(seed=args.seed, max_tasks=args.max_tasks)

    output_dir.mkdir(parents=True)
    manifest = build_phase2a_manifest(
        run_id=output_dir.name,
        base_sha=base_sha,
        phase1_freeze_sha=phase1_freeze_sha,
        builder_commit_sha=builder_commit_sha,
        source_paths=adapter.source_paths,
        forbidden_paths=tuple(Path(path).expanduser() for path in args.forbidden_source_path),
        image_root_config=adapter.image_root_config,
        image_root_resolved_path=adapter.image_root,
        stage=adapter.stage,
        seed=result.seed,
        max_tasks=result.max_tasks,
        total_source_rows=result.total_source_rows,
        accepted_before_sampling=result.accepted_before_sampling,
        selected_after_sampling=result.selected_after_sampling,
        output_root=output_dir,
        extra={"forbidden_index_records": forbidden_index.record_count},
    )
    write_manifest(manifest, output_dir / "manifest.json")

    serialized_entries = [entry.to_dict() for entry in result.entries]
    _write_jsonl(output_dir / "source_index.jsonl", serialized_entries)
    _write_jsonl(
        output_dir / "rejected.jsonl",
        [entry for entry in serialized_entries if entry["status"] != "accepted"],
    )

    # Constructing the existing budget object creates the ledger schema without
    # reserving a request.  The sentinel scope is never used for generation.
    ledger_path = output_dir / "teacher_requests.sqlite3"
    TeacherRequestBudget(
        ledger_path,
        "__phase2a_preflight__",
        limit=32,
    )
    _close_empty_ledger(ledger_path)

    summary: dict[str, object] = {
        "status": "preflight_complete",
        "run_dir": str(output_dir),
        "total_source_rows": result.total_source_rows,
        "accepted_before_sampling": result.accepted_before_sampling,
        "selected_after_sampling": result.selected_after_sampling,
        "counts": result.counts,
        "seed": result.seed,
        "max_tasks": result.max_tasks,
        "forbidden_index_records": forbidden_index.record_count,
        "teacher_slots_consumed": 0,
        "snapshots_created": 0,
        "trajectories_created": 0,
        "export_candidates": 0,
        "final_sft_rows": 0,
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, SourcePreflightError, RunDirectoryConflict) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
