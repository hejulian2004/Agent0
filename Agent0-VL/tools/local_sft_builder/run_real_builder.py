"""Local SFT builder entry points.

Two modes, mutually exclusive and both required to be chosen explicitly:

``--dry-run``
    Phase 2A behaviour, unchanged: source preflight plus deterministic
    sampling.  Writes exactly four files and can never consume a Teacher slot.

``--generate``
    Phase 2B: real source -> real Teacher -> natural Solver rollout -> frozen
    Validator -> Projector -> exact dedup -> ``messages + images`` rows.

Both modes refuse to overwrite an existing run directory, and neither mode ever
resumes an earlier run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .budget import TeacherRequestBudget
from .pipeline import validate_project_deduplicate
from .projector import write_export_jsonl
from .run_manifest import (
    build_phase2a_manifest,
    build_phase2b_manifest,
    write_manifest,
)
from .runtime import RealSandboxRuntime, UpstreamSandboxRunner
from .schema import ExportCandidate
from .snapshot_store import SnapshotStore
from .source_adapter import (
    ForbiddenSourceIndex,
    RealSourceAdapter,
    SourcePreflightError,
    SourcePreflightResult,
)
from .source_guard import SourceLeakageGuard
from .teacher_backend import (
    DEFAULT_API_KEY_ENV,
    TeacherBackend,
    SamplingConfig,
)
from .trajectory_builder import RealTrajectoryBuilder


EXPECTED_BASE_SHA = "f775b5101e62fe92976831adf4a21a38fcc0a767"

# Rows whose frozen partition record has no usable ``resolved_revision`` cannot
# form a (dataset, revision, original_id) identity and are therefore absent from
# the forbidden index.  They stay blocked structurally: only
# ``formal_sft_stage*`` rows are normalized, and ``SourceLeakageGuard`` rejects
# any ``split != train`` or ``usage_partition`` outside the SFT partitions.
FORBIDDEN_COVERAGE_NOTE = (
    "Rows in the frozen partitions whose record has no usable "
    "resolved_revision cannot form a (dataset, revision, original_id) identity "
    "and are excluded from this index. They remain blocked structurally: only "
    "formal_sft_stage* rows are normalized, and SourceLeakageGuard rejects any "
    "split != train or usage_partition outside {sft_stage1, sft_stage2}."
)

# The Teacher must be an OpenAI-compatible server (``vllm serve``).  Its API
# root is ``/v1``, and ``TeacherBackend`` appends ``/chat/completions``, so the
# ``/v1`` here is load-bearing: omitting it produces a 404, not a request.
DEFAULT_TEACHER_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_TEACHER_MODEL = "qwen3.8-27b"


class RunDirectoryConflict(RuntimeError):
    """Raised when a run would overwrite an existing directory."""


class TeacherNotConfigured(ValueError):
    """Raised when the Teacher credential environment variable is unset."""


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


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _frozen_base_sha(value: str) -> str:
    if value != EXPECTED_BASE_SHA:
        raise argparse.ArgumentTypeError(
            f"must equal the frozen Phase 2A base SHA {EXPECTED_BASE_SHA}"
        )
    return value


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    dict(value),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _candidate_to_dict(candidate: ExportCandidate) -> dict[str, Any]:
    return {
        "task_id": candidate.task_id,
        "trajectory_id": candidate.trajectory_id,
        "source_record_id": candidate.source_record_id,
        "stage": candidate.stage,
        "messages": [dict(message) for message in candidate.messages],
        "images": list(candidate.images),
        "image_content_hashes": list(candidate.image_content_hashes),
        "validation": candidate.validation.to_dict(),
    }


@dataclass(frozen=True)
class _Prepared:
    """Everything both modes compute before they diverge."""

    output_dir: Path
    base_sha: str
    phase1_freeze_sha: str
    builder_commit_sha: str
    forbidden_index: ForbiddenSourceIndex
    adapter: RealSourceAdapter
    preflight: SourcePreflightResult

    def serialized_entries(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.preflight.entries]

    def rejected_entries(self) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self.serialized_entries()
            if entry["status"] != "accepted"
        ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local SFT builder. --dry-run performs source preflight only; "
            "--generate runs the real Teacher rollout and writes export rows."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Phase 2A preflight only; consumes no Teacher request slot.",
    )
    mode.add_argument(
        "--generate",
        action="store_true",
        help="Phase 2B real generation; consumes Teacher request slots.",
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
    # Teacher connection.  The credential itself is only ever read from the
    # environment; --teacher-api-key-env names the variable, never a value.
    parser.add_argument("--teacher-base-url", default=DEFAULT_TEACHER_BASE_URL)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=_positive_int, default=1024)
    parser.add_argument("--request-timeout", type=_positive_float, default=300.0)
    # Rollout.
    parser.add_argument("--max-reasoning-steps", type=_positive_int, default=8)
    parser.add_argument("--budget-limit", type=_positive_int, default=32)
    parser.add_argument("--sandbox-timeout", type=_positive_float)
    return parser


def _prepare(args: argparse.Namespace) -> _Prepared:
    repo_root = (
        Path(args.repo_root).expanduser().resolve() if args.repo_root else _repo_root()
    )
    output_dir = Path(args.output_run_dir).expanduser().resolve()
    if output_dir.exists():
        raise RunDirectoryConflict(
            f"refusing to overwrite existing run directory: {output_dir}"
        )

    if args.base_sha != EXPECTED_BASE_SHA:
        raise ValueError(
            f"base SHA must equal the frozen commit {EXPECTED_BASE_SHA}"
        )
    base_sha = _git_revision(repo_root, args.base_sha)
    if base_sha != EXPECTED_BASE_SHA:
        raise ValueError(
            f"resolved base SHA does not equal the frozen commit {EXPECTED_BASE_SHA}"
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
    preflight = adapter.preflight(seed=args.seed, max_tasks=args.max_tasks)
    return _Prepared(
        output_dir=output_dir,
        base_sha=base_sha,
        phase1_freeze_sha=phase1_freeze_sha,
        builder_commit_sha=builder_commit_sha,
        forbidden_index=forbidden_index,
        adapter=adapter,
        preflight=preflight,
    )


def _run_dry_run(args: argparse.Namespace) -> dict[str, object]:
    prepared = _prepare(args)
    output_dir = prepared.output_dir
    output_dir.mkdir(parents=True)

    manifest = build_phase2a_manifest(
        run_id=output_dir.name,
        base_sha=prepared.base_sha,
        phase1_freeze_sha=prepared.phase1_freeze_sha,
        builder_commit_sha=prepared.builder_commit_sha,
        source_paths=prepared.adapter.source_paths,
        forbidden_paths=tuple(
            Path(path).expanduser() for path in args.forbidden_source_path
        ),
        image_root_config=prepared.adapter.image_root_config,
        image_root_resolved_path=prepared.adapter.image_root,
        stage=prepared.adapter.stage,
        seed=prepared.preflight.seed,
        max_tasks=prepared.preflight.max_tasks,
        total_source_rows=prepared.preflight.total_source_rows,
        accepted_before_sampling=prepared.preflight.accepted_before_sampling,
        selected_after_sampling=prepared.preflight.selected_after_sampling,
        output_root=output_dir,
        extra={"forbidden_index_records": prepared.forbidden_index.record_count},
    )
    write_manifest(manifest, output_dir / "manifest.json")

    _write_jsonl(output_dir / "source_index.jsonl", prepared.serialized_entries())
    _write_jsonl(output_dir / "rejected.jsonl", prepared.rejected_entries())

    # Constructing the existing budget object creates the ledger schema without
    # reserving a request.  The sentinel scope is never used for generation.
    ledger_path = output_dir / "teacher_requests.sqlite3"
    TeacherRequestBudget(
        ledger_path,
        "__phase2a_preflight__",
        limit=32,
    )

    summary: dict[str, object] = {
        "status": "preflight_complete",
        "run_dir": str(output_dir),
        "total_source_rows": prepared.preflight.total_source_rows,
        "accepted_before_sampling": prepared.preflight.accepted_before_sampling,
        "selected_after_sampling": prepared.preflight.selected_after_sampling,
        "counts": prepared.preflight.counts,
        "seed": prepared.preflight.seed,
        "max_tasks": prepared.preflight.max_tasks,
        "forbidden_index_records": prepared.forbidden_index.record_count,
        "teacher_slots_consumed": 0,
        "snapshots_created": 0,
        "trajectories_created": 0,
        "export_candidates": 0,
        "final_sft_rows": 0,
    }
    return summary


def _empty_request_counts() -> dict[str, int]:
    return {
        "consumed_slot": 0,
        "pending": 0,
        "successful": 0,
        "timeout": 0,
        "parse_failed": 0,
        "other_failed": 0,
    }


def _run_generate(args: argparse.Namespace) -> dict[str, object]:
    prepared = _prepare(args)
    output_dir = prepared.output_dir

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.request_timeout,
        seed=args.seed,
    )
    credential_env = args.teacher_api_key_env
    if not os.environ.get(credential_env):
        raise TeacherNotConfigured(
            f"{credential_env} is not set; export the Teacher credential in the "
            "environment before running --generate (it is never read from a file "
            "or a command-line value)"
        )

    backend = TeacherBackend(
        base_url=args.teacher_base_url,
        model=args.teacher_model,
        api_key_env=credential_env,
        default_sampling=sampling,
    )
    sandbox_runner = UpstreamSandboxRunner(timeout_seconds=args.sandbox_timeout)
    sandbox = RealSandboxRuntime(sandbox=sandbox_runner)
    ledger_path = output_dir / "teacher_requests.sqlite3"

    output_dir.mkdir(parents=True)
    snapshot_store = SnapshotStore(output_dir / "snapshots")
    builder = RealTrajectoryBuilder(
        backend=backend,
        budget_db=str(ledger_path),
        sandbox_runtime=sandbox,
        source_guard=SourceLeakageGuard(),
        max_reasoning_steps=args.max_reasoning_steps,
        budget_limit=args.budget_limit,
        sampling=sampling,
    )

    trajectories: list[Any] = []
    audit_records: list[Any] = []
    candidates: list[ExportCandidate] = []
    source_decisions: dict[str, Any] = {}
    request_counts = _empty_request_counts()
    natural_success = 0
    natural_failure = 0
    snapshots_written = 0

    for task in prepared.preflight.selected_tasks:
        result = builder.build_task(task)
        source_decisions[task.task_id] = result.source_guard
        if result.root_snapshot is not None:
            snapshot_store.write(result.root_snapshot)
            snapshots_written += 1
        trajectories.extend(result.trajectories)
        audit_records.extend(result.audit_records)
        candidates.extend(result.candidates)

        stats = result.budget_stats
        request_counts["consumed_slot"] += stats.consumed_slot
        request_counts["pending"] += stats.pending
        request_counts["successful"] += stats.successful
        request_counts["timeout"] += stats.timeout
        request_counts["parse_failed"] += stats.parse_failed
        request_counts["other_failed"] += stats.other_failed
        if result.rollout is not None:
            if result.rollout.completed:
                natural_success += 1
            else:
                natural_failure += 1

    # The frozen gate runs once more over every collected trajectory so the
    # exported rows and the recorded decisions come from a single pass.
    pipeline = validate_project_deduplicate(
        trajectories,
        source_decisions,
        max_reasoning_steps=args.max_reasoning_steps,
    )
    rows_before_dedup = list(pipeline.rows_before_dedup)
    rows_after_dedup = list(pipeline.rows_after_dedup)

    _write_jsonl(output_dir / "source_index.jsonl", prepared.serialized_entries())
    _write_jsonl(output_dir / "rejected.jsonl", prepared.rejected_entries())
    _write_jsonl(
        output_dir / "trajectories.jsonl",
        [trajectory.to_dict() for trajectory in trajectories],
    )
    _write_jsonl(
        output_dir / "audit_records.jsonl",
        [record.to_dict() for record in audit_records],
    )
    _write_jsonl(
        output_dir / "validation_decisions.jsonl",
        [decision.to_dict() for decision in pipeline.validation_decisions],
    )
    _write_jsonl(
        output_dir / "export_candidates.jsonl",
        [_candidate_to_dict(candidate) for candidate in candidates],
    )
    _write_jsonl(
        output_dir / "stage1.jsonl",
        [row.to_dict() for row in rows_before_dedup if row.stage == "sft_stage1"],
    )
    _write_jsonl(
        output_dir / "stage2.jsonl",
        [row.to_dict() for row in rows_before_dedup if row.stage == "sft_stage2"],
    )
    write_export_jsonl(rows_after_dedup, output_dir / "final_dedup.jsonl")

    manifest = build_phase2b_manifest(
        run_id=output_dir.name,
        base_sha=prepared.base_sha,
        phase1_freeze_sha=prepared.phase1_freeze_sha,
        builder_commit_sha=prepared.builder_commit_sha,
        source_paths=prepared.adapter.source_paths,
        forbidden_paths=tuple(
            Path(path).expanduser() for path in args.forbidden_source_path
        ),
        image_root_config=prepared.adapter.image_root_config,
        image_root_resolved_path=prepared.adapter.image_root,
        stage=prepared.adapter.stage,
        seed=prepared.preflight.seed,
        max_tasks=prepared.preflight.max_tasks,
        total_source_rows=prepared.preflight.total_source_rows,
        accepted_before_sampling=prepared.preflight.accepted_before_sampling,
        selected_after_sampling=prepared.preflight.selected_after_sampling,
        output_root=output_dir,
        teacher=backend.describe(),
        teacher_sampling=sampling.to_dict(),
        builder=builder.describe(),
        teacher_request_counts=request_counts,
        natural_success=natural_success,
        natural_failure=natural_failure,
        export_candidates=len(candidates),
        dedup_dropped=len(rows_before_dedup) - len(rows_after_dedup),
        final_sft_rows=len(rows_after_dedup),
        snapshot_count=snapshots_written,
        trajectory_count=len(trajectories),
        audit_record_count=len(audit_records),
        extra={
            "forbidden_index_records": prepared.forbidden_index.record_count,
            "forbidden_coverage_note": FORBIDDEN_COVERAGE_NOTE,
            "sandbox": sandbox_runner.describe(),
        },
    )
    write_manifest(manifest, output_dir / "manifest.json")

    summary: dict[str, object] = {
        "status": "generation_complete",
        "run_dir": str(output_dir),
        "total_source_rows": prepared.preflight.total_source_rows,
        "accepted_before_sampling": prepared.preflight.accepted_before_sampling,
        "selected_after_sampling": prepared.preflight.selected_after_sampling,
        "counts": prepared.preflight.counts,
        "seed": prepared.preflight.seed,
        "max_tasks": prepared.preflight.max_tasks,
        "forbidden_index_records": prepared.forbidden_index.record_count,
        "teacher_slots_consumed": request_counts["consumed_slot"],
        "teacher_request_counts": request_counts,
        "snapshots_created": snapshots_written,
        "trajectories_created": len(trajectories),
        "audit_records": len(audit_records),
        "natural_success": natural_success,
        "natural_failure": natural_failure,
        "export_candidates": len(candidates),
        "dedup_dropped": len(rows_before_dedup) - len(rows_after_dedup),
        "final_sft_rows": len(rows_after_dedup),
    }
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.generate:
        return _run_generate(args)
    return _run_dry_run(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except (
        OSError,
        ValueError,
        SourcePreflightError,
        RunDirectoryConflict,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
