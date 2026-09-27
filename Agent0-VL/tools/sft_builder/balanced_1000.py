"""Single-stage balanced light reproduction: eight sources x 111 + one x 112.

Uses prepared training candidates from nine SFT sources. Final Swift rows have only messages and images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from .rebuild_paper_1000 import SOURCES, _validation_questions, _digest

SOURCES_ALL = tuple(SOURCES)
WORK = Path("data/sft/balanced_1000_v2")
FINAL = Path("data/sft/large/mixed_balanced_1000.jsonl")


def prepare(root: Path, seed: int, sources=None) -> None:
    _assert_builder_stopped()
    work = root / WORK
    work.mkdir(parents=True, exist_ok=True)
    quotas = {name: 112 if i == 0 else 111 for i, name in enumerate(SOURCES_ALL)}
    excluded = _validation_questions(root)
    seen = set(excluded)
    manifest = {"seed": seed, "mode": "single_stage", "quotas": quotas,
                "selection": "equal source quotas; seeded rank within prepared train candidates",
                "teacher_model": "qwen3.8-27b", "candidate_rows": {},
                "note": "Nine paper SFT sources; ChartQA and ThinkLite remain RL-only."}
    if sources:
        manifest_path = work / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("Full prepare is required before selective prepare")
        manifest = json.loads(manifest_path.read_text())
        if manifest["seed"] != seed:
            raise RuntimeError("Selective prepare must use the existing seed")
        for other in SOURCES_ALL:
            if other in sources:
                continue
            for line in (work / f"{other}_candidates.jsonl").read_text().splitlines():
                row = json.loads(line)
                seen.add(_digest(" ".join(row["question"].replace("<image>", "").casefold().split())))
    for source in SOURCES_ALL:
        if sources and source not in sources:
            continue
        print(f"[balanced-sft] scanning {source}", flush=True)
        records = []
        # Re-normalize raw train rows so reference/choice fixes are reflected.
        # The prior question index already enforces train/validation partitioning.
        import sqlite3
        from .rebuild_paper_1000 import _rows
        from .sources import _normalize_row, SourceFormatError
        source_path = root / SOURCES[source]
        connection = sqlite3.connect(root / "data/sft/rebuild_paper_1000/source_index.sqlite")
        ordinals = {item[0] for item in connection.execute(
            "SELECT ordinal FROM samples WHERE source=? ORDER BY rank LIMIT ?",
            (source, max(5000, quotas[source] * 50)))}
        connection.close()
        for ordinal, (row, base, file) in enumerate(_rows(source, source_path)):
            if ordinal not in ordinals:
                continue
            try:
                sample = _normalize_row(row, base, file, ordinal, source, 2)
            except SourceFormatError:
                continue
            # Strict generation cannot accept missing-reference candidates.
            if not sample.get("ground_truth"):
                continue
            records.append(sample)
        records.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{source}:{row['task_id']}".encode()).hexdigest())
        selected = []
        for row in records:
            key = _digest(" ".join(row["question"].replace("<image>", "").casefold().split()))
            if key in seen:
                continue
            seen.add(key)
            row["stage"] = 2
            selected.append(row)
        if len(selected) < quotas[source]:
            raise RuntimeError(f"{source}: {len(selected)} candidates, need {quotas[source]}")
        output = work / f"{source}_candidates.jsonl"
        temporary = output.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(output)
        manifest["candidate_rows"][source] = len(selected)
        print(f"[balanced-sft] prepared {source}: candidates={len(selected)}, final_quota={quotas[source]}", flush=True)
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def preflight(root: Path) -> None:
    work = root / WORK
    manifest_path = work / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Fresh strict candidates required: bash scripts/rebuild-balanced-data.sh prepare")
    manifest = json.loads(manifest_path.read_text())
    if sum(manifest["quotas"].values()) != 1000:
        raise RuntimeError("Source quotas do not total 1000")
    for source, quota in manifest["quotas"].items():
        count = 0
        with (work / f"{source}_candidates.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("ground_truth"):
                    raise RuntimeError(f"{source}: candidate has no reference answer; rebuild candidates")
                if any(not Path(image).is_file() and not image.startswith(("data:", "http:", "https:")) for image in row["images"]):
                    raise RuntimeError(f"{source}: missing candidate image")
                count += 1
        if count < quota or count != manifest["candidate_rows"][source]:
            raise RuntimeError(f"{source}: candidate count/manifest mismatch")
        state_path = work / f"{source}.jsonl.state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            config = state.get("run_config", {})
            if config.get("row_audit_version") != 2 or config.get("solver_verification_flow_version") != 6 or config.get("keep_unverified") is not False:
                raise RuntimeError(f"{source}: incompatible old generation state; stop the running builder, then run reset-generation")
        print(f"[balanced-sft] preflight {source}: referenced_candidates={count} quota={quota}", flush=True)
    print("[balanced-sft] data preflight passed", flush=True)


def build(root: Path, args) -> None:
    preflight(root)
    work = root / WORK
    manifest = json.loads((work / "manifest.json").read_text())
    outputs = []
    for source, quota in manifest["quotas"].items():
        output = work / f"{source}.jsonl"
        existing_state_path = Path(str(output) + ".state.json")
        if existing_state_path.exists():
            state = json.loads(existing_state_path.read_text())
            if state.get("target_reached"):
                config = state["run_config"]
                expected = {
                    "source_sha256": hashlib.sha256((work / f"{source}_candidates.jsonl").read_bytes()).hexdigest(),
                    "system_prompt_sha256": hashlib.sha256((root / "scripts/prompt.txt").read_bytes()).hexdigest(),
                    "teacher_model": args.teacher_model,
                    "teacher_max_tokens": args.teacher_max_tokens,
                    "max_reasoning_steps": args.max_reasoning_steps,
                }
                if any(config.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"{source}: completed output configuration changed; refusing reuse")
                from .merge_sft import audit_record
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                if len(rows) != quota or any(audit_record(row, 2, allow_stage2_images=True) for row in rows):
                    raise RuntimeError(f"{source}: completed output fails quota or audit")
                print(f"[balanced-sft] reuse completed {source}: {quota} rows; teacher calls skipped", flush=True)
                outputs.append(output)
                continue
        command = [sys.executable, "-m", "tools.sft_builder.build_stream",
                   "--source", "normalized", "--source-path", str(work / f"{source}_candidates.jsonl"),
                   "--stage", "2", "--quality-profile", "stage2", "--output", str(output),
                   "--max-tasks", str(manifest["candidate_rows"][source]),
                   "--target-exported", str(quota), "--concurrency", str(args.concurrency),
                   "--batch-size", str(args.concurrency), "--teacher-base-url", args.teacher_url,
                   "--max-reasoning-steps", str(args.max_reasoning_steps), "--teacher-max-tokens", str(args.teacher_max_tokens), "--teacher-model", args.teacher_model, "--teacher-timeout", str(args.teacher_timeout), "--repo-root", str(root)]
        if Path(str(output) + ".state.json").is_file():
            command.append("--resume")
        print(f"[balanced-sft] generating {source}: quota={quota}", flush=True)
        subprocess.run(command, cwd=root, check=True)
        state = json.loads(Path(str(output) + ".state.json").read_text())
        if not state.get("target_reached") or state["exported_rows"] != quota:
            raise RuntimeError(f"{source} failed to reach {quota} verified rows; inspect state file")
        outputs.append(output)
    final = root / FINAL
    subprocess.run([sys.executable, "-m", "tools.sft_builder.merge_sft", "--stage", "2",
                    "--allow-stage2-images", "--input", *map(str, outputs), "--output", str(final)],
                   cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "tools.sft_builder.validate_sft", "--stage", "2",
                    "--allow-stage2-images", "--input", str(final), "--expected-rows", "1000"],
                   cwd=root, check=True)
    rows = [json.loads(line) for line in final.read_text().splitlines()]
    random.Random(manifest["seed"]).shuffle(rows)
    final.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest.update(final_output=str(final), final_rows=len(rows), teacher_model=args.teacher_model,
                    prompt_sha256=hashlib.sha256((root / "scripts/prompt.txt").read_bytes()).hexdigest())
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[balanced-sft] complete: {final} rows={len(rows)}", flush=True)


def _assert_builder_stopped() -> None:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"tools.sft_builder.build_stream" in args or (
            b"tools.sft_builder.balanced_1000" in args and b"build" in args
        ):
            raise RuntimeError("Stop the active builder with Ctrl+C before changing candidates or generation state")

def reset_generation(root: Path, sources=None) -> None:
    """Archive only selected source outputs; candidates are untouched."""
    from datetime import datetime
    _assert_builder_stopped()
    work = root / WORK
    archive = work / ("generation_archive_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    paths = [p for source in (sources or SOURCES_ALL) for p in
             (work / f"{source}.jsonl", work / f"{source}.jsonl.state.json") if p.exists()]
    if paths:
        archive.mkdir(parents=True)
        for path in paths:
            path.rename(archive / path.name)
    print(f"[balanced-sft] archived {len(paths)} generation files to {archive}; candidates preserved", flush=True)


def restore_completed(root: Path, sources) -> None:
    import shutil
    from .merge_sft import audit_record
    _assert_builder_stopped()
    work = root / WORK
    manifest = json.loads((work / "manifest.json").read_text())
    prompt_hash = hashlib.sha256((root / "scripts/prompt.txt").read_bytes()).hexdigest()
    planned = []
    for source in sources:
        destination = work / f"{source}.jsonl"
        if destination.exists() or Path(str(destination) + ".state.json").exists():
            raise RuntimeError(f"{source}: current files exist; refuse overwrite")
        candidates_hash = hashlib.sha256((work / f"{source}_candidates.jsonl").read_bytes()).hexdigest()
        for archive in sorted(work.glob("generation_archive_*"), reverse=True):
            state_path = archive / f"{source}.jsonl.state.json"
            output = archive / f"{source}.jsonl"
            if not state_path.exists() or not output.exists():
                continue
            state = json.loads(state_path.read_text())
            config = state.get("run_config", {})
            if not state.get("target_reached") or config.get("source_sha256") != candidates_hash or config.get("system_prompt_sha256") != prompt_hash:
                continue
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            if len(rows) != manifest["quotas"][source] or any(audit_record(row, 2, allow_stage2_images=True) for row in rows):
                continue
            planned.append((source, output, state_path, destination))
            break
        else:
            raise RuntimeError(f"{source}: no compatible audited completed archive")
    for source, output, state_path, destination in planned:
        shutil.copy2(output, destination)
        shutil.copy2(state_path, Path(str(destination) + ".state.json"))
        print(f"[balanced-sft] restored completed {source} from {output.parent}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "preflight", "build", "all", "reset-generation", "restore-completed"])
    parser.add_argument("--sources", nargs="+", choices=SOURCES_ALL)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-reasoning-steps", type=int, default=16)
    parser.add_argument("--teacher-max-tokens", type=int, default=8192)
    parser.add_argument("--teacher-timeout", type=float, default=600.0)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--teacher-model", default="qwen3.8-27b")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    if args.action == "restore-completed":
        restore_completed(root, args.sources or SOURCES_ALL)
    if args.action == "reset-generation":
        reset_generation(root, args.sources)
    if args.action in ("prepare", "all"):
        prepare(root, args.seed, args.sources)
    if args.action == "preflight":
        preflight(root)
    if args.action in ("build", "all"):
        build(root, args)


if __name__ == "__main__":
    main()
