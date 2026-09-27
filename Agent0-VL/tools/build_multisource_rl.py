"""Build reproducible, multimodal RL training data from local train splits.

The repository's earlier RL files were derived from the We-Math testmini
artifact.  This builder deliberately uses only rows assigned to the RL
partition by ``global_partition_v1`` and whose source split is train.  It
creates a balanced candidate pool, then two disjoint subsets for the external
correctness warm-up and the SERC/GRPO phase.

The local license review marks ChartQA as verified.  ArxivQA and ThinkLite-VL
are retained as research candidates, but their rows carry
``license_status=manual_review_required`` in ``extra_info`` and the manifest;
the script never silently presents them as license-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SEED = 20260922
DATASETS = ("chartqa", "arxivqa", "thinklite")


def proportional_quotas(available: Mapping[str, int], target: int) -> dict[str, int]:
    """Apportion by downloaded train-row counts, reserving one per source."""
    if target < len(available) or any(count < 1 for count in available.values()):
        raise ValueError("Every RL source needs at least one row and one slot")
    if sum(available.values()) < target:
        raise ValueError("Not enough downloaded training rows for the requested subset")
    total = sum(available.values())
    exact = {name: target * count / total for name, count in available.items()}
    result = {name: max(1, int(exact[name])) for name in available}
    while sum(result.values()) > target:
        name = min((name for name in result if result[name] > 1),
                   key=lambda name: (exact[name] - result[name], name))
        result[name] -= 1
    while sum(result.values()) < target:
        name = max((name for name in result if result[name] < available[name]),
                   key=lambda name: (exact[name] - result[name], name))
        result[name] += 1
    return result


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _rank(seed: int, namespace: str, value: Mapping[str, Any]) -> str:
    stable_id = str(value.get("source_record_id") or value.get("original_id") or value)
    return hashlib.sha256(f"{seed}:{namespace}:{stable_id}".encode("utf-8")).hexdigest()


def _rl_key(row: Mapping[str, Any]) -> str:
    payload = {
        "prompt": row["prompt"],
        "ground_truth": row["reward_model"]["ground_truth"],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _question_key(value: str) -> str:
    return hashlib.sha256(" ".join(_strip_image_marker(value).casefold().split()).encode("utf-8")).hexdigest()


def _excluded_questions(root: Path) -> set[str]:
    excluded: set[str] = set()
    validation = root / "data/rl/validation_10_rebuilt.parquet"
    if validation.is_file():
        for row in pq.read_table(validation, columns=["prompt"]).to_pylist():
            prompt = row.get("prompt") or []
            if prompt:
                excluded.add(_question_key(str(prompt[0].get("content", ""))))
    for sft_path in (root / "data/sft/large/mixed_balanced_1000.jsonl",
                     root / "data/sft/large/stage1_500.jsonl",
                     root / "data/sft/large/stage2_500.jsonl"):
        if not sft_path.is_file():
            continue
        for row in _read_jsonl(sft_path):
            first = next((message for message in row.get("messages", [])
                          if message.get("role") == "user"), None)
            if first:
                excluded.add(_question_key(str(first.get("content", ""))))
    return excluded


def _strip_image_marker(text: str) -> str:
    text = text.strip()
    while text.lower().startswith("<image>"):
        text = text[len("<image>") :].lstrip(" \n")
    return text


def _prompt(question: str, options: Iterable[str] = ()) -> str:
    content = "<image>\n" + _strip_image_marker(question)
    clean_options = [_clean_text(option) for option in options if _clean_text(option)]
    if clean_options:
        content += "\n\nOptions:\n" + "\n".join(clean_options)
    return content


def _arxiv_label(value: Any) -> str | None:
    """Return a choice letter, rejecting synthetic placeholder labels."""

    label = _clean_text(value)
    match = re.match(r"^([A-H])(?:\s*(?:[).:]|$))", label, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def _metadata(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASETS}
    for row in _read_jsonl(path):
        dataset = str(row.get("dataset", ""))
        if dataset not in grouped:
            continue
        if row.get("usage_partition") != "rl":
            continue
        if str(row.get("split", "")).lower() != "train":
            continue
        official_split = str(row.get("official_split", "")).lower()
        if official_split not in {"train", "train_inferred"}:
            continue
        grouped[dataset].append(row)
    for dataset in DATASETS:
        grouped[dataset].sort(key=lambda row: (_rank(DEFAULT_SEED, dataset, row), str(row)))
    return grouped


def _downloaded_train_counts(root: Path) -> dict[str, int]:
    """Count source training records before global SFT/RL partitioning."""
    chartqa = root / "data/raw/.staging/chartqa-full-proxy/ChartQA Dataset/train"
    arxivqa = root / "data/raw/.staging/arxivqa-probe/arxivqa.jsonl"
    thinklite = root / "data/raw/.staging/thinklite-proxy/ThinkLite-VL-70k.parquet"
    return {
        "chartqa": sum(len(json.loads((chartqa / name).read_text(encoding="utf-8")))
                       for name in ("train_human.json", "train_augmented.json")),
        "arxivqa": sum(bool(line.strip()) for line in arxivqa.open(encoding="utf-8")),
        "thinklite": pq.ParquetFile(thinklite).metadata.num_rows,
    }


def _select_metadata(
    grouped: Mapping[str, list[dict[str, Any]]],
    quotas: Mapping[str, int],
    seed: int,
    reserve: int = 256,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        rows = sorted(
            grouped.get(dataset, []),
            key=lambda row: (_rank(seed, f"candidate:{dataset}", row), str(row)),
        )
        need = int(quotas.get(dataset, 0))
        if len(rows) < need:
            raise ValueError(f"{dataset} has {len(rows)} train RL candidates; need {need}")
        selected[dataset] = rows[: need + reserve]
    return selected


def _common_extra(
    metadata: Mapping[str, Any],
    *,
    source_row_index: int,
    source_image: str,
) -> dict[str, Any]:
    return {
        "index": int(source_row_index),
        "source_id": str(metadata.get("source_record_id") or metadata.get("original_id", "")),
        "source_dataset": str(metadata.get("dataset", "")),
        "source_row_index": int(source_row_index),
        "source_image": source_image,
        "split": str(metadata.get("split", "train")),
        "official_split": str(metadata.get("official_split", "")),
        "validation_only": False,
        "formal_training_eligible": bool(metadata.get("formal_training_eligible", False)),
        "license_status": str(metadata.get("license_status", "unknown")),
    }


def _row(
    *,
    question: str,
    answer: str,
    image_bytes: bytes,
    data_source: str,
    extra_info: dict[str, Any],
    options: Iterable[str] = (),
) -> dict[str, Any]:
    if not question or not answer or not image_bytes:
        raise ValueError("RL row is missing question, answer, or image bytes")
    return {
        "prompt": [{"role": "user", "content": _prompt(question, options)}],
        "images": [{"bytes": bytes(image_bytes)}],
        "reward_model": {"ground_truth": answer},
        "data_source": data_source,
        "extra_info": extra_info,
    }


def _load_chartqa(
    project_root: Path,
    metadata_rows: list[dict[str, Any]],
    quota: int,
    seed: int,
) -> list[dict[str, Any]]:
    dataset_root = project_root / "data/raw/.staging/chartqa-full-proxy/ChartQA Dataset"
    members = {
        "ChartQA Dataset/train/train_human.json": dataset_root / "train/train_human.json",
        "ChartQA Dataset/train/train_augmented.json": dataset_root / "train/train_augmented.json",
    }
    raw = {member: json.loads(path.read_text(encoding="utf-8")) for member, path in members.items()}
    ranked = sorted(
        metadata_rows,
        key=lambda row: (_rank(seed, "row:chartqa", row), str(row)),
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metadata in ranked:
        member = str(metadata.get("raw_member", ""))
        index = int(metadata["row_index"])
        source_row = raw.get(member, [])[index]
        image_name = _clean_text(source_row.get("imgname"))
        image_path = dataset_root / "train/png" / image_name
        if not image_path.is_file():
            continue
        query = _clean_text(source_row.get("query"))
        answer = _clean_text(source_row.get("label"))
        if not query or not answer:
            continue
        extra = _common_extra(metadata, source_row_index=index, source_image=image_name)
        row = _row(
            question=query,
            answer=answer,
            image_bytes=image_path.read_bytes(),
            data_source="chartqa",
            extra_info=extra,
        )
        key = _rl_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) == quota:
            return result
    raise RuntimeError(f"ChartQA yielded {len(result)} unique rows; required {quota}")


def _load_arxivqa(
    project_root: Path,
    metadata_rows: list[dict[str, Any]],
    quota: int,
    seed: int,
) -> list[dict[str, Any]]:
    dataset_root = project_root / "data/raw/.staging/arxivqa-probe"
    raw_path = dataset_root / "arxivqa.jsonl"
    ranked = sorted(
        metadata_rows,
        key=lambda row: (_rank(seed, "row:arxivqa", row), str(row)),
    )
    # Read only a bounded ranked prefix.  The reserve absorbs malformed
    # GPT-generated placeholder labels present in the upstream file.
    wanted = {int(row["row_index"]) for row in ranked[: quota + 256]}
    raw: dict[int, dict[str, Any]] = {}
    with raw_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                raw[index] = json.loads(line)
            if len(raw) == len(wanted):
                break

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metadata in ranked:
        index = int(metadata["row_index"])
        source_row = raw.get(index)
        if source_row is None:
            continue
        answer = _arxiv_label(source_row.get("label"))
        question = _clean_text(source_row.get("question"))
        image_name = _clean_text(source_row.get("image"))
        image_path = dataset_root / image_name
        if answer is None or not question or not image_path.is_file():
            continue
        extra = _common_extra(metadata, source_row_index=index, source_image=image_name)
        row = _row(
            question=question,
            answer=answer,
            image_bytes=image_path.read_bytes(),
            data_source="arxivqa",
            extra_info=extra,
            options=source_row.get("options") or (),
        )
        key = _rl_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) == quota:
            return result
    raise RuntimeError(
        f"ArxivQA yielded {len(result)} unique answerable rows from the ranked prefix; "
        f"required {quota}"
    )


def _load_thinklite(
    project_root: Path,
    metadata_rows: list[dict[str, Any]],
    quota: int,
    seed: int,
) -> list[dict[str, Any]]:
    raw_path = project_root / "data/raw/.staging/thinklite-proxy/ThinkLite-VL-70k.parquet"
    ranked = sorted(
        metadata_rows,
        key=lambda row: (_rank(seed, "row:thinklite", row), str(row)),
    )
    candidate_metadata = ranked[: quota + 256]
    wanted = {int(row["row_index"]) for row in candidate_metadata}
    raw: dict[int, dict[str, Any]] = {}
    parquet_file = pq.ParquetFile(raw_path)
    for batch in parquet_file.iter_batches(
        columns=["image", "problem", "answer", "id", "ground_truth"],
        batch_size=512,
    ):
        for source_row in batch.to_pylist():
            index = int(source_row.get("id"))
            if index in wanted:
                raw[index] = source_row
        if len(raw) == len(wanted):
            break

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for metadata in ranked:
        index = int(metadata["row_index"])
        source_row = raw.get(index)
        if source_row is None:
            continue
        image_bytes = source_row.get("image")
        question = _clean_text(source_row.get("problem"))
        answer = _clean_text(source_row.get("ground_truth")) or _clean_text(source_row.get("answer"))
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            continue
        if not question or not answer:
            continue
        extra = _common_extra(metadata, source_row_index=index, source_image=f"row:{index}")
        row = _row(
            question=question,
            answer=answer,
            image_bytes=bytes(image_bytes),
            data_source="thinklite",
            extra_info=extra,
        )
        key = _rl_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) == quota:
            return result
    raise RuntimeError(f"ThinkLite yielded {len(result)} unique rows; required {quota}")


def _write_dataset(rows: list[dict[str, Any]], path: Path, preview_path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty RL dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with preview_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            prompt = "\n".join(str(message.get("content", "")) for message in row["prompt"])
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "source_id": row["extra_info"]["source_id"],
                        "data_source": row["data_source"],
                        "prompt": prompt,
                        "ground_truth": row["reward_model"]["ground_truth"],
                        "image_count": len(row["images"]),
                        "license_status": row["extra_info"]["license_status"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _select_rows(
    rows: Iterable[dict[str, Any]],
    quotas: Mapping[str, int],
    *,
    seed: int,
    namespace: str,
    excluded: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded or set()
    selected: list[dict[str, Any]] = []
    for dataset in DATASETS:
        candidates = [
            row
            for row in rows
            if row["data_source"] == dataset and _rl_key(row) not in excluded
        ]
        candidates.sort(
            key=lambda row: (
                _rank(seed, f"{namespace}:{dataset}", row["extra_info"]),
                row["extra_info"]["source_id"],
            )
        )
        quota = int(quotas.get(dataset, 0))
        if len(candidates) < quota:
            raise ValueError(f"{dataset} has {len(candidates)} available rows; need {quota} for {namespace}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda row: (_rank(seed, namespace, row["extra_info"]), _rl_key(row)))
    return selected


def _manifest(
    *,
    output: Path,
    rows: list[dict[str, Any]],
    seed: int,
    source: str,
    excluded_count: int = 0,
) -> dict[str, Any]:
    source_counts = Counter(str(row["data_source"]) for row in rows)
    license_counts = Counter(str(row["extra_info"]["license_status"]) for row in rows)
    return {
        "output": str(output),
        "source": source,
        "rows": len(rows),
        "seed": seed,
        "selection": "seeded SHA-256 rank, downloaded-source-proportional, duplicate-free",
        "data_sources": dict(sorted(source_counts.items())),
        "license_status": dict(sorted(license_counts.items())),
        "validation_only_rows": sum(bool(row["extra_info"]["validation_only"]) for row in rows),
        "manual_review_rows": sum(
            not bool(row["extra_info"]["formal_training_eligible"]) for row in rows
        ),
        "excluded_rows_from_previous_subset": excluded_count,
        "split_policy": "train and train_inferred only; test/testmini excluded",
        "schema_fields": ["prompt", "images", "reward_model", "data_source", "extra_info"],
    }


def build(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    metadata_path = (root / args.metadata).resolve() if not args.metadata.is_absolute() else args.metadata
    grouped = _metadata(metadata_path)
    available = {dataset: len(grouped[dataset]) for dataset in DATASETS}
    downloaded = _downloaded_train_counts(root)
    formal_quotas = proportional_quotas(downloaded, 200)
    warmup_quotas = proportional_quotas(downloaded, 200)
    pool_quotas = proportional_quotas(downloaded, 1000)
    for dataset in DATASETS:
        pool_quotas[dataset] = max(pool_quotas[dataset], formal_quotas[dataset] + warmup_quotas[dataset])
    # A pool larger than 1,000 is harmless; each phase still receives 200 rows.
    if args.preflight_only:
        required = [
            root / "data/raw/.staging/chartqa-full-proxy/ChartQA Dataset/train/train_human.json",
            root / "data/raw/.staging/chartqa-full-proxy/ChartQA Dataset/train/train_augmented.json",
            root / "data/raw/.staging/arxivqa-probe/arxivqa.jsonl",
            root / "data/raw/.staging/thinklite-proxy/ThinkLite-VL-70k.parquet",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing RL training source files: " + ", ".join(missing))
        print(json.dumps({"downloaded_train_rows": downloaded,
                          "eligible_rl_partition_rows": available, "pool_quotas": pool_quotas,
                          "formal_quotas": formal_quotas, "warmup_quotas": warmup_quotas},
                         ensure_ascii=False, indent=2))
        return
    if sum(pool_quotas.values()) < sum(formal_quotas.values()) + sum(warmup_quotas.values()):
        raise ValueError("RL pool is smaller than formal plus warm-up subsets")

    selected_metadata = _select_metadata(grouped, pool_quotas, args.seed)
    loaders = {
        "chartqa": _load_chartqa,
        "arxivqa": _load_arxivqa,
        "thinklite": _load_thinklite,
    }
    pool: list[dict[str, Any]] = []
    for dataset in DATASETS:
        print(f"[multisource-rl] loading {dataset}: target={pool_quotas[dataset]}", flush=True)
        pool.extend(
            loaders[dataset](
                root,
                selected_metadata[dataset],
                pool_quotas[dataset],
                args.seed,
            )
        )
    excluded_questions = _excluded_questions(root)
    pool = [row for row in pool
            if _question_key(str(row["prompt"][0]["content"])) not in excluded_questions]
    # A different ground truth must not make the same question a new sample.
    seen_questions: set[str] = set()
    unique_pool = []
    for row in pool:
        question_key = _question_key(str(row["prompt"][0]["content"]))
        if question_key not in seen_questions:
            seen_questions.add(question_key)
            unique_pool.append(row)
    pool = unique_pool
    pool_keys = {_rl_key(row) for row in pool}
    if len(pool_keys) != len(pool):
        raise AssertionError("RL pool contains duplicate prompt/ground-truth rows")
    if len(pool) < 400:
        raise AssertionError(f"only {len(pool)} RL rows remain after validation/SFT exclusion")

    formal = _select_rows(pool, formal_quotas, seed=args.seed, namespace="formal")
    formal_keys = {_rl_key(row) for row in formal}
    warmup = _select_rows(
        pool,
        warmup_quotas,
        seed=args.seed,
        namespace="warmup",
        excluded=formal_keys,
    )
    warmup_keys = {_rl_key(row) for row in warmup}
    if formal_keys & warmup_keys:
        raise AssertionError("formal and warm-up RL subsets overlap")

    pool_path = root / args.pool_output
    formal_path = root / args.formal_output
    warmup_path = root / args.warmup_output
    _write_dataset(pool, pool_path, root / args.pool_preview)
    _write_dataset(formal, formal_path, root / args.formal_preview)
    _write_dataset(warmup, warmup_path, root / args.warmup_preview)

    manifests = {
        "pool": _manifest(output=pool_path, rows=pool, seed=args.seed, source=str(metadata_path)),
        "formal": _manifest(
            output=formal_path,
            rows=formal,
            seed=args.seed,
            source=str(pool_path),
        ),
        "warmup": _manifest(
            output=warmup_path,
            rows=warmup,
            seed=args.seed,
            source=str(pool_path),
            excluded_count=len(formal),
        ),
    }
    manifest_path = root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": str(metadata_path),
                "seed": args.seed,
                "downloaded_train_rows_by_source": downloaded,
                "eligible_rl_partition_rows_by_source": available,
                "quota_basis": "downloaded training record counts before SFT/RL partition, largest remainder with one per source",
                "excluded_validation_sft_question_count": len(excluded_questions),
                "excluded_eval_sources": ["mathverse", "mathvista", "wemath_validation"],
                "datasets": manifests,
                "formal_warmup_disjoint": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, info in manifests.items():
        print(json.dumps({"name": name, **info}, ensure_ascii=False, indent=2))
    print(json.dumps({"manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/global_partition_v1/assigned_rl.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--pool-output", type=Path, default=Path("data/rl/rl_multisource_pool_1000.parquet"))
    parser.add_argument("--formal-output", type=Path, default=Path("data/rl/rl_200_multisource.parquet"))
    parser.add_argument("--warmup-output", type=Path, default=Path("data/rl/rl_warmup_200_multisource.parquet"))
    parser.add_argument("--pool-preview", type=Path, default=Path("data/rl/rl_multisource_pool_1000.preview.jsonl"))
    parser.add_argument("--formal-preview", type=Path, default=Path("data/rl/rl_200_multisource.preview.jsonl"))
    parser.add_argument("--warmup-preview", type=Path, default=Path("data/rl/rl_warmup_200_multisource.preview.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/rl/rl_multisource.manifest.json"))
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
