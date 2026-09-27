"""Rebuild two sequential 500-row SFT sets from paper-listed train sources.

Run ``preflight`` first. ``prepare`` indexes downloaded train records and
materializes deterministic teacher candidates; ``build`` generates Stage 1,
audits it, then generates Stage 2. No published per-source paper ratios exist:
quotas use the downloaded, deduplicated train-record counts at each stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .sources import (
    SourceFormatError,
    _extract_question,
    _iter_file_rows,
    _iter_geoqa_rows,
    _iter_geometry_rows,
    _normalize_row,
)


SOURCES = {
    "geometry3k": "data/raw/.staging/geometry3k-probe/raw/train",
    "geoqa": "data/raw/.staging/geoqa-official-proxy/GeoQA3",
    "mulberry": "data/raw/.staging/mulberry-proxy/mulberry_sft.json",
    "llava_ov_image": "data/raw/.staging/llava-ov-image/train",
    "mm_rlhf": "data/raw/.staging/mm-rlhf/train",
    "smr": "data/raw/.staging/smr/train",
    "mmeureka": "data/raw/.staging/mmeureka-proxy/dataset.jsonl",
    "retool": "data/raw/.staging/retool-proxy/train_2000.parquet",
    "arxivqa": "data/raw/.staging/arxivqa-probe/arxivqa.jsonl",
}
STAGES = {
    "geometry3k": (1,), "geoqa": (1,), "llava_ov_image": (1,),
    "mm_rlhf": (1,), "smr": (1,), "arxivqa": (1,),
    "mulberry": (1, 2), "retool": (1, 2), "mmeureka": (2,),
}
OUTPUT_ROOT = Path("data/sft/rebuild_paper_1000")
FINAL_ROOT = Path("data/sft/large")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rows(source: str, path: Path) -> Iterator[tuple[dict[str, Any], Path, Path]]:
    if source == "geometry3k":
        yield from _iter_geometry_rows(path)
    elif source == "geoqa":
        yield from _iter_geoqa_rows(path, split="train")
    else:
        for row, base, file in _iter_file_rows(path):
            relative = file.relative_to(path) if path.is_dir() else Path(file.name)
            if any(re.search(r"(^|[_-])(test|testmini|dev|validation|val)([_-]|$)",
                             part.lower()) for part in relative.parts):
                continue
            yield row, base, file


def _quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    if not counts or any(value < 1 for value in counts.values()) or target < len(counts):
        raise ValueError("Every paper source must have train rows and at least one quota slot")
    if sum(counts.values()) < target:
        raise ValueError(f"Only {sum(counts.values())} unique train records; need {target}")
    total = sum(counts.values())
    exact = {name: target * count / total for name, count in counts.items()}
    result = {name: max(1, int(exact[name])) for name in counts}
    while sum(result.values()) > target:
        name = min((n for n in counts if result[n] > 1),
                   key=lambda n: (exact[n] - result[n], n))
        result[name] -= 1
    while sum(result.values()) < target:
        name = max((n for n in counts if result[n] < counts[n]),
                   key=lambda n: (exact[n] - result[n], n))
        result[name] += 1
    return result


def _source_paths(root: Path, overrides: list[str]) -> dict[str, Path]:
    paths = {name: root / path for name, path in SOURCES.items()}
    for item in overrides:
        name, separator, value = item.partition("=")
        if not separator or name not in SOURCES or not value:
            raise ValueError(f"Expected --source NAME=PATH; names: {', '.join(SOURCES)}")
        path = Path(value).expanduser()
        paths[name] = path if path.is_absolute() else root / path
    return paths


def preflight(root: Path, paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    for required in (root / "scripts/prompt.txt", root / ".venv/bin/python"):
        if not required.is_file():
            missing.append(str(required))
    if missing:
        raise RuntimeError("Missing paper SFT inputs:\n" + "\n".join(missing))
    def includes_eval_split(path: Path) -> bool:
        # Inspect the dataset path, not workspace ancestors such as
        # ``Agent0-dev-codex``, which otherwise falsely matches ``dev``.
        parts = path.relative_to(root).parts if path.is_relative_to(root) else path.parts[-2:]
        return any(
            re.search(r"(^|[_-])(test|testmini|dev|validation|val)([_-]|$)", part.lower())
            for part in parts
        )

    bad_paths = [f"{name}: {path}" for name, path in paths.items()
                 if includes_eval_split(path)]
    if bad_paths:
        raise RuntimeError("Evaluation paths cannot be used for SFT:\n" + "\n".join(bad_paths))
    print("Nine SFT source paths and the main-format prompt are present.")


def _excluded_arxiv_ids(root: Path) -> set[str]:
    metadata = root / "data/processed/global_partition_v1/assigned_rl.jsonl"
    if not metadata.is_file():
        raise RuntimeError(f"Required cross-stage arXivQA partition is missing: {metadata}")
    result: set[str] = set()
    with metadata.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") == "arxivqa":
                result.add(str(row.get("original_id", "")))
    return result


def _validation_questions(root: Path) -> set[str]:
    excluded: set[str] = set()
    for path in (root / "data/sft/validation").glob("*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for message in row.get("messages", []):
                    if message.get("role") == "user" and not str(message.get("content", "")).startswith("[Code Execution Result]"):
                        excluded.add(_digest(" ".join(str(message.get("content", "")).replace("<image>", "").casefold().split())))
                        break
    rl_path = root / "data/rl/validation_10_rebuilt.parquet"
    if rl_path.is_file():
        import pyarrow.parquet as pq
        for row in pq.read_table(rl_path, columns=["prompt"]).to_pylist():
            prompt = row.get("prompt") or []
            if prompt:
                excluded.add(_digest(" ".join(str(prompt[0].get("content", "")).replace("<image>", "").casefold().split())))
    return excluded


def prepare(root: Path, paths: dict[str, Path], seed: int, candidate_factor: int) -> None:
    preflight(root, paths)
    if candidate_factor < 1:
        raise ValueError("candidate_factor must be positive")
    work = root / OUTPUT_ROOT
    work.mkdir(parents=True, exist_ok=True)
    database = work / "source_index.sqlite"
    if database.exists():
        database.unlink()
    excluded_arxiv = _excluded_arxiv_ids(root)
    validation_questions = _validation_questions(root)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE samples (source TEXT, stage INTEGER, ordinal INTEGER, rank TEXT, question_hash TEXT UNIQUE)")
    connection.execute("CREATE INDEX lookup ON samples(source, stage, rank)")
    rejected = Counter()
    raw_counts = Counter()
    try:
        for source, path in paths.items():
            for ordinal, (row, _base, _file) in enumerate(_rows(source, path)):
                raw_counts[source] += 1
                split = str(row.get("split", "train")).lower()
                if split not in {"train", "training", "train_inferred"}:
                    rejected[f"{source}:non_train"] += 1
                    continue
                source_id = str(row.get("id") or row.get("pid") or row.get("task_id") or ordinal)
                if source == "arxivqa" and source_id in excluded_arxiv:
                    rejected["arxivqa:rl_partition"] += 1
                    continue
                question = _extract_question(row)
                if not question:
                    rejected[f"{source}:no_question"] += 1
                    continue
                stages = STAGES[source]
                stage = stages[int(_digest(f"{seed}:{source}:{source_id}"), 16) % len(stages)]
                key = _digest(" ".join(question.casefold().split()))
                if key in validation_questions:
                    rejected[f"{source}:validation_overlap"] += 1
                    continue
                rank = _digest(f"{seed}:{source}:{source_id}")
                try:
                    connection.execute("INSERT INTO samples VALUES (?, ?, ?, ?, ?)",
                                       (source, stage, ordinal, rank, key))
                except sqlite3.IntegrityError:
                    rejected[f"{source}:duplicate_question"] += 1
            connection.commit()

        counts = Counter({
            (source, stage): count for source, stage, count in connection.execute(
                "SELECT source, stage, COUNT(*) FROM samples GROUP BY source, stage")
        })
        manifest: dict[str, Any] = {
            "seed": seed, "teacher_model": "qwen3.8-27b",
            "quota_basis": "downloaded unique train record counts before teacher filtering",
            "source_paths": {name: str(path) for name, path in paths.items()},
            "downloaded_rows": dict(raw_counts), "rejected": dict(rejected), "stages": {},
        }
        for stage in (1, 2):
            stage_counts = {source: counts.get((source, stage), 0)
                            for source in SOURCES if stage in STAGES[source]}
            quotas = _quotas(stage_counts, 500)
            manifest["stages"][str(stage)] = {
                "unique_train_rows": stage_counts, "quotas": quotas,
            }
            for source, quota in quotas.items():
                take = min(stage_counts[source], max(1000, quota * candidate_factor))
                selected = {ordinal for (ordinal,) in connection.execute(
                    "SELECT ordinal FROM samples WHERE source=? AND stage=? ORDER BY rank LIMIT ?",
                    (source, stage, take))}
                output = work / f"stage{stage}_{source}_candidates.jsonl"
                normalized = 0
                with output.open("w", encoding="utf-8") as handle:
                    for ordinal, (row, base, file) in enumerate(_rows(source, paths[source])):
                        if ordinal not in selected:
                            continue
                        try:
                            sample = _normalize_row(row, base, file, ordinal, source, stage)
                        except SourceFormatError:
                            rejected[f"{source}:normalize_failed"] += 1
                            continue
                        handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        normalized += 1
                if normalized < quota:
                    raise RuntimeError(f"{source} stage {stage}: only {normalized} usable candidates; need {quota}")
                manifest["stages"][str(stage)].setdefault("candidate_rows", {})[source] = normalized
        manifest["rejected"] = dict(rejected)
        (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        connection.close()


def build(root: Path, teacher_url: str, teacher_model: str, concurrency: int) -> None:
    work = root / OUTPUT_ROOT
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    python = str(root / ".venv/bin/python")
    env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY for the local Qwen teacher before build")
    for stage in (1, 2):
        inputs = []
        final_counts: dict[str, int] = {}
        for source, quota in manifest["stages"][str(stage)]["quotas"].items():
            candidates = work / f"stage{stage}_{source}_candidates.jsonl"
            output = work / f"stage{stage}_{source}.jsonl"
            state = Path(str(output) + ".state.json")
            command = [python, "-m", "tools.sft_builder.build_stream",
                       "--source", source, "--source-path", str(candidates),
                       "--stage", str(stage), "--quality-profile", f"stage{stage}",
                       "--output", str(output), "--max-tasks",
                       str(manifest["stages"][str(stage)]["candidate_rows"][source]),
                       "--target-exported", str(quota), "--concurrency", str(concurrency),
                       "--batch-size", str(max(1, concurrency)),
                       "--teacher-base-url", teacher_url, "--teacher-model", teacher_model,
                       "--repo-root", str(root)]
            if state.exists():
                command.append("--resume")
            subprocess.run(command, cwd=root, env=env, check=True)
            status = json.loads(state.read_text(encoding="utf-8"))
            if not status.get("target_reached") or status.get("exported_rows") != quota:
                raise RuntimeError(f"{source} Stage {stage} did not reach quota {quota}")
            final_counts[source] = int(status["exported_rows"])
            inputs.append(output)
        final = root / FINAL_ROOT / f"stage{stage}_500.jsonl"
        final.parent.mkdir(parents=True, exist_ok=True)
        merge = [python, "-m", "tools.sft_builder.merge_sft", "--stage", str(stage),
                 "--input", *(str(path) for path in inputs), "--output", str(final),
                 "--manifest", str(final.with_suffix(".manifest.json"))]
        if stage == 2:
            merge.append("--allow-stage2-images")
        subprocess.run(merge, cwd=root, env=env, check=True)
        validate = [python, "-m", "tools.sft_builder.validate_sft", "--stage", str(stage),
                    "--input", str(final), "--expected-rows", "500"]
        if stage == 2:
            validate.append("--allow-stage2-images")
        subprocess.run(validate, cwd=root, env=env, check=True)
        manifest["stages"][str(stage)]["final_rows"] = final_counts
        manifest["stages"][str(stage)]["final_output"] = str(final)
        manifest["teacher_model"] = teacher_model
        (work / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Stage {stage} complete: {final}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "prepare", "build", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--candidate-factor", type=int, default=50)
    parser.add_argument("--teacher-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--teacher-model", default="qwen3.8-27b")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = _source_paths(root, args.source)
    if args.action == "preflight":
        preflight(root, paths)
    elif args.action == "prepare":
        prepare(root, paths, args.seed, args.candidate_factor)
    elif args.action == "build":
        build(root, args.teacher_url, args.teacher_model, args.concurrency)
    else:
        prepare(root, paths, args.seed, args.candidate_factor)
        build(root, args.teacher_url, args.teacher_model, args.concurrency)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, SourceFormatError) as exc:
        raise SystemExit(str(exc)) from exc
