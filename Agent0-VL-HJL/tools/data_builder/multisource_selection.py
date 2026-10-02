"""Build disjoint RL subsets from three train sources and three user-authorized
public evaluation releases repurposed for training. Preserve original splits.
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
TRAIN_DATASETS = ("chartqa", "arxivqa", "thinklite")
EVAL_TRAIN_PATHS = {
    "mathverse": "data/raw/.staging/mathverse-proxy/testmini.parquet",
    "mathvista": "data/raw/.staging/mathvista-probe/testmini-00000-of-00001-725687bf7a18d64b.parquet",
    "wemath": "data/raw/.staging/wemath-probe/official/testmini.parquet",
}
DATASETS = TRAIN_DATASETS + tuple(EVAL_TRAIN_PATHS)


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
    value = value.split("[Tool Runtime]", 1)[0]
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
    for dataset in TRAIN_DATASETS:
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


def _rl_group_key(row):
    extra = row['extra_info']
    problem = extra.get('source_problem_id')
    return f"{row['data_source']}:{problem}" if problem else _rl_key(row)


def _answer_kind(row: Mapping[str, Any]) -> str:
    """Conservative read-time preference; reference length alone is insufficient."""
    from tools.data_builder.sft_quality import StrictAnswerJudge
    def answer_options(task):
        return task.get("options") or StrictAnswerJudge.extract_options(task.get("question", ""))

    question = str(row['prompt'][0]['content'])
    extra = row['extra_info']
    options = answer_options({'question': question, 'options': extra.get('options')})
    if options:
        return 'multiple_choice'
    answer = str(row['reward_model']['ground_truth']).strip()
    if re.search(r'(?i)\b(explain|describe|discuss|justify|prove|write an? essay|give reasons)\b|解释|描述|论述|证明|说明理由', question):
        return 'open_ended'
    if re.fullmatch(r'[\s\d.,+\-−%/$€£°():=\\{}^_a-zA-Z]+', answer) and re.search(r'\d', answer) and len(answer) <= 80:
        return 'numeric'
    if len(answer) <= 80 and len(answer.split()) <= 8 and not re.search(r'[\n.!?。！？;；]', answer):
        return 'short_answer'
    return 'open_ended'


def _eval_metadata(root, dataset):
    path = root / EVAL_TRAIN_PATHS[dataset]
    columns = {'mathverse': ['problem_index', 'sample_index'],
               'mathvista': ['pid'], 'wemath': ['ID', 'question number']}[dataset]
    records = pq.read_table(path, columns=columns).to_pylist()
    blocked = set()
    if dataset == 'wemath':
        validation = root / 'data/rl/validation_10_rebuilt.parquet'
        if validation.is_file():
            blocked = {str(r['extra_info'].get('source_id', ''))
                       for r in pq.read_table(validation, columns=['extra_info']).to_pylist()}
    selected = {}
    for index, record in enumerate(records):
        problem = str(record.get('problem_index', record.get('pid', record.get('ID'))))
        if problem in blocked:
            continue
        # Keep one version/subquestion per problem to prevent split leakage.
        if problem not in selected:
            selected[problem] = {
                'dataset': dataset, 'row_index': index,
                'source_record_id': f'{dataset}:{index}', 'source_problem_id': problem,
                'split': 'testmini', 'official_split': 'testmini',
                'formal_training_eligible': True,
                'license_status': 'unknown',
                'repurposed_for_training': True,
            }
    return list(selected.values())


def _load_eval_training(root, metadata_rows, quota, seed):
    dataset = metadata_rows[0]['dataset']
    metadata = {int(r['row_index']): r for r in metadata_rows}
    selected = []
    offset = 0
    for batch in pq.ParquetFile(root / EVAL_TRAIN_PATHS[dataset]).iter_batches(batch_size=64):
        for local, source in enumerate(batch.to_pylist()):
            index = offset + local
            if index not in metadata:
                continue
            question = _clean_text(source.get('question'))
            answer = _clean_text(source.get('answer'))
            options = source.get('choices') or []
            if dataset == 'mathverse':
                image = source.get('image') or {}
            elif dataset == 'mathvista':
                image = source.get('decoded_image') or {}
                options = [f'{chr(65 + i)}. {option}' for i, option in enumerate(options)]
            else:
                image = source.get('image_path') or {}
                # Preserve the original inline option text; the shared judge
                # receives it through answer_options' raw option representation.
                option_text = _clean_text(source.get('option'))
                if option_text:
                    question += '\n\nOptions:\n' + option_text
            image_bytes = image.get('bytes') if isinstance(image, dict) else None
            if not question or not answer or not image_bytes:
                continue
            extra = _common_extra(metadata[index], source_row_index=index, source_image=f'row:{index}')
            extra.update(source_problem_id=metadata[index]['source_problem_id'],
                         repurposed_for_training=True, original_split='testmini',
                         future_benchmark_use=False)
            if options:
                extra['options'] = options
            if dataset == 'mathvista':
                extra.update(unit=source.get('unit'), precision=source.get('precision'))
            selected.append(_row(question=question, answer=answer, image_bytes=bytes(image_bytes),
                                 data_source=dataset, extra_info=extra, options=options))
        offset += batch.num_rows
    selected.sort(key=lambda row: _rank(seed, dataset, row['extra_info']))
    if len(selected) < quota:
        raise RuntimeError(f'{dataset} yielded {len(selected)} valid rows; need {quota}')
    return selected


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
            if row["data_source"] == dataset and _rl_group_key(row) not in excluded
        ]
        candidates.sort(
            key=lambda row: (
                {'multiple_choice': 0, 'numeric': 1, 'short_answer': 2}.get(row['extra_info'].get('answer_kind'), 3),
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
        "split_policy": "train/train_inferred plus user-authorized MathVerse/MathVista/WeMath testmini repurposed for training; original split retained; existing validation excluded",
        "schema_fields": ["prompt", "images", "reward_model", "data_source", "extra_info"],
    }
