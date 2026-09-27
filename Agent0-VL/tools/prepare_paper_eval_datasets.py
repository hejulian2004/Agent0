#!/usr/bin/env python3
"""Download and normalize the seven Agent0-VL main-table evaluation sets.

The output Parquet schema is compatible with verl.utils.dataset.RLHFDataset:
``prompt`` is a chat list, ``images`` contains image bytes, and
``reward_model`` stores benchmark-specific scoring metadata.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING = PROJECT_ROOT / "data" / "raw" / ".staging"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "agent0_vl_paper_main"

EXPECTED_ROWS = {
    "mathverse": 3940,
    "mathvision": 3040,
    "mathvista": 1000,
    "wemath": 1740,
    "hallusionbench": 1129,
    "chartqa": 2500,
    "mmmu": 900,
}

PROMPT_SCHEMA = pa.list_(pa.struct([
    ("role", pa.string()),
    ("content", pa.string()),
]))
IMAGE_SCHEMA = pa.list_(pa.struct([
    ("bytes", pa.binary()),
    ("path", pa.string()),
]))
REWARD_SCHEMA = pa.struct([
    ("ground_truth", pa.string()),
    ("benchmark", pa.string()),
    ("score_type", pa.string()),
    ("answers", pa.list_(pa.string())),
    ("options", pa.list_(pa.string())),
    ("precision", pa.float64()),
])
EXTRA_SCHEMA = pa.struct([
    ("index", pa.int64()),
    ("sample_id", pa.string()),
    ("split", pa.string()),
    ("metadata_json", pa.string()),
])
OUTPUT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("data_source", pa.string()),
    ("prompt", PROMPT_SCHEMA),
    ("images", IMAGE_SCHEMA),
    ("reward_model", REWARD_SCHEMA),
    ("extra_info", EXTRA_SCHEMA),
])


def _parse_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null", "[]"}):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    text = str(value).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if item is not None]
    return [text]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _image_bytes(value: Any, *, label: str) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, Image.Image):
        from io import BytesIO

        out = BytesIO()
        value.convert("RGB").save(out, format="PNG")
        value = out.getvalue()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise ValueError(f"{label}: image is not embedded as nonempty bytes")
    raw = bytes(value)
    try:
        with Image.open(__import__("io").BytesIO(raw)) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label}: unreadable image payload ({exc})") from exc
    return raw


def _read_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=64):
            yield from batch.to_pylist()


def _answer_and_score_type(benchmark: str, answer: Any, answer_type: Any = None) -> tuple[str, str]:
    truth = _as_text(answer)
    if benchmark in {"mathverse", "mathvision", "wemath", "mmmu"} and re.fullmatch(r"[A-H]", truth, flags=re.I):
        return truth.upper(), "multiple_choice"
    if benchmark == "hallusionbench":
        return ("yes" if truth == "1" else "no" if truth == "0" else truth), "yes_no"
    if benchmark == "chartqa":
        return truth, "chartqa"
    if benchmark == "mathvista":
        kind = _as_text(answer_type).lower()
        if kind in {"integer", "float", "number"}:
            return truth, "numeric"
        if kind == "choice":
            return truth, "multiple_choice"
        return truth, "math_or_text"
    if benchmark == "mathvision":
        return truth, "math_or_text"
    return truth, "math_or_text"


def _format_options(options: list[str]) -> str:
    if not options:
        return ""
    lines = []
    for idx, option in enumerate(options):
        label = chr(ord("A") + idx)
        clean = re.sub(r"^\s*[A-H][.)：:]\s*", "", option, flags=re.I)
        lines.append(f"({label}) {clean}")
    return "\nOptions:\n" + "\n".join(lines)


def _replace_numbered_images(text: str, image_map: dict[int, bytes], *, sample_id: str) -> tuple[str, list[bytes]]:
    ordered: list[bytes] = []
    used: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        image = image_map.get(idx)
        if image is None:
            return f"[image {idx} unavailable]"
        ordered.append(image)
        used.add(idx)
        return "<image>"

    text = re.sub(r"<\s*image\s*(\d+)\s*>", replace, text, flags=re.I)
    for idx, image in sorted(image_map.items()):
        if idx not in used:
            text += f"\nImage {idx}: <image>"
            ordered.append(image)
    return text, ordered


def _record(
    *,
    benchmark: str,
    split: str,
    sample_id: Any,
    question: str,
    answer: Any,
    images: list[bytes],
    options: list[str] | None = None,
    answer_type: Any = None,
    precision: Any = None,
    metadata: dict[str, Any] | None = None,
    image_tokens_are_numbered: bool = False,
) -> dict[str, Any]:
    options = options or []
    sample_id = _as_text(sample_id)
    question = _as_text(question)
    if not sample_id or not question:
        raise ValueError(f"{benchmark}: row lacks an id or question")
    truth, score_type = _answer_and_score_type(benchmark, answer, answer_type)
    if not truth:
        raise ValueError(f"{benchmark}/{sample_id}: row lacks a labeled answer")

    content = question
    if options and not re.search(r"\b(?:choices|options)\s*:", question, flags=re.I):
        content += _format_options(options)
    if score_type == "multiple_choice":
        content += "\nGive the final answer as the option letter."
    elif score_type == "yes_no":
        content += "\nAnswer yes or no."
    else:
        content += "\nGive a concise final answer."

    if image_tokens_are_numbered:
        image_map = {idx + 1: image for idx, image in enumerate(images)}
        content, images = _replace_numbered_images(content, image_map, sample_id=sample_id)
    elif images:
        generic_count = len(re.findall(r"<image>", content, flags=re.I))
        if generic_count == 0:
            content = "\n".join(["<image>"] * len(images)) + "\n" + content
        elif generic_count != len(images):
            raise ValueError(
                f"{benchmark}/{sample_id}: prompt has {generic_count} image markers but {len(images)} images"
            )

    marker_count = len(re.findall(r"<image>", content, flags=re.I))
    if marker_count != len(images):
        raise ValueError(
            f"{benchmark}/{sample_id}: prompt has {marker_count} image markers but {len(images)} images"
        )

    try:
        number_precision = float(precision) if precision is not None else None
    except (TypeError, ValueError):
        number_precision = None
    if number_precision is not None and number_precision < 0:
        number_precision = None

    return {
        "id": f"{benchmark}:{sample_id}",
        "data_source": benchmark,
        "prompt": [{"role": "user", "content": content}],
        "images": [{"bytes": image, "path": None} for image in images],
        "reward_model": {
            "ground_truth": truth,
            "benchmark": benchmark,
            "score_type": score_type,
            "answers": [truth],
            "options": options,
            "precision": number_precision,
        },
        "extra_info": {
            "index": -1,
            "sample_id": sample_id,
            "split": split,
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, default=str),
        },
    }


def _load_mathverse() -> list[dict[str, Any]]:
    path = STAGING / "mathverse-proxy" / "testmini.parquet"
    rows = []
    for row in _read_rows([path]):
        image = _image_bytes(row.get("image"), label=f"MathVerse/{row.get('sample_index')}")
        question = _as_text(row.get("question_for_eval") or row.get("question"))
        options = [f"{label}. {value.strip()}" for label, value in re.findall(
            r"(?:^|\n)\s*([A-H])\s*[:.)：]\s*([^\n]+)", question, flags=re.I
        )]
        rows.append(_record(
            benchmark="mathverse", split="testmini", sample_id=row.get("sample_index"),
            question=question, answer=row.get("answer"),
            images=[image] if image else [], options=options, answer_type=row.get("question_type"),
            metadata={"problem_index": row.get("problem_index"), "problem_version": row.get("problem_version"),
                      "question_type": row.get("question_type"), "metadata": row.get("metadata")},
        ))
    return rows


def _load_mathvision() -> list[dict[str, Any]]:
    path = next((STAGING / "mathvision-probe" / "data").glob("test-*.parquet"))
    rows = []
    for row in _read_rows([path]):
        image = _image_bytes(row.get("decoded_image") or row.get("image"), label=f"MathVision/{row.get('id')}")
        options = _parse_list(row.get("options"))
        # MathVision's numbered tags point to panels within one composite
        # page image. They are not separate image payloads, so retain their
        # references as text and attach the one bundled page image once.
        question = re.sub(
            r"<\s*image\s*(\d+)\s*>", r"Image \1", _as_text(row.get("question")), flags=re.I
        )
        rows.append(_record(
            benchmark="mathvision", split="test", sample_id=row.get("id"),
            question=question, answer=row.get("answer"), images=[image] if image else [],
            options=options, answer_type="multiple-choice" if options else "free-form",
            metadata={"level": row.get("level"), "subject": row.get("subject")},
        ))
    return rows


def _load_mathvista() -> list[dict[str, Any]]:
    path = next((STAGING / "mathvista-probe").glob("testmini-*.parquet"))
    rows = []
    for row in _read_rows([path]):
        image = _image_bytes(row.get("decoded_image") or row.get("image"), label=f"MathVista/{row.get('pid')}")
        rows.append(_record(
            benchmark="mathvista", split="testmini", sample_id=row.get("pid"),
            question=row.get("query") or row.get("question"), answer=row.get("answer"),
            images=[image] if image else [], options=_parse_list(row.get("choices")),
            answer_type=row.get("answer_type"), precision=row.get("precision"),
            metadata={"question_type": row.get("question_type"), "unit": row.get("unit"),
                      "answer_type": row.get("answer_type"), "metadata": row.get("metadata")},
        ))
    return rows


def _extract_wemath_parquet() -> Path:
    archive = STAGING / "wemath-probe" / "We-Math.zip"
    output = STAGING / "wemath-probe" / "official" / "testmini.parquet"
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        parquet_name = next(name for name in source.namelist()
                            if name.startswith("We-Math/data/testmini-") and name.endswith(".parquet"))
        output.write_bytes(source.read(parquet_name))
    return output


def _load_wemath() -> list[dict[str, Any]]:
    path = _extract_wemath_parquet()
    rows = []
    for row in _read_rows([path]):
        image = _image_bytes(row.get("image_path"), label=f"WeMath/{row.get('ID')}")
        option_text = _as_text(row.get("option"))
        options = [part.strip() for part in re.split(r";\s*(?=[A-H][.)：:])", option_text) if part.strip()]
        rows.append(_record(
            # ``ID`` identifies a problem family and ``key`` identifies its
            # variant; together they form the source's per-row unique key.
            benchmark="wemath", split="testmini",
            sample_id=f"{row.get('ID')}:{row.get('key')}",
            question=row.get("question"), answer=row.get("answer"), images=[image] if image else [],
            options=options, answer_type="multiple-choice",
            metadata={"knowledge_concept": row.get("knowledge concept"), "question_number": row.get("question number")},
        ))
    return rows


def _load_hallusionbench() -> list[dict[str, Any]]:
    paths = [
        STAGING / "hallusionbench-probe" / "data" / "image-00000-of-00001.parquet",
        STAGING / "hallusionbench-probe" / "data" / "non_image-00000-of-00001.parquet",
    ]
    rows = []
    for row in _read_rows(paths):
        image = _image_bytes(row.get("image"), label=f"HallusionBench/{row.get('question_id')}")
        sample_id = "-".join(_as_text(row.get(key)) for key in
                             ("category", "subcategory", "set_id", "figure_id", "question_id"))
        rows.append(_record(
            benchmark="hallusionbench", split="image+non_image", sample_id=sample_id,
            question=row.get("question"), answer=row.get("gt_answer"), images=[image] if image else [],
            metadata={"category": row.get("category"), "subcategory": row.get("subcategory"),
                      "visual_input": row.get("visual_input"), "set_id": row.get("set_id"),
                      "figure_id": row.get("figure_id"), "gt_answer_details": row.get("gt_answer_details")},
        ))
    return rows


def _load_chartqa() -> list[dict[str, Any]]:
    archive = STAGING / "chartqa-full-proxy" / "ChartQA Dataset.zip"
    rows = []
    with zipfile.ZipFile(archive) as source:
        for split_file in ("test_human.json", "test_augmented.json"):
            split = split_file.removesuffix(".json")
            annotations = json.loads(source.read(f"ChartQA Dataset/test/{split_file}"))
            for idx, row in enumerate(annotations):
                image_path = f"ChartQA Dataset/test/png/{row['imgname']}"
                image = _image_bytes(source.read(image_path), label=f"ChartQA/{split}/{idx}")
                rows.append(_record(
                    benchmark="chartqa", split="test", sample_id=f"{split}:{idx}",
                    question=row.get("query"), answer=row.get("label"), images=[image],
                    metadata={"chart_split": split, "image_name": row.get("imgname")},
                ))
    return rows


def _normalize_mmmu_rows() -> list[dict[str, Any]]:
    """Build MMMU separately so choice metadata and numbered images stay aligned."""
    root = STAGING / "mmmu-probe" / "official"
    paths = sorted(root.glob("*" + "/validation-*.parquet"))
    rows = []
    for row in _read_rows(paths):
        image_map = {}
        for idx in range(1, 8):
            raw = _image_bytes(row.get(f"image_{idx}"), label=f"MMMU/{row.get('id')}/image_{idx}")
            if raw:
                image_map[idx] = raw
        options = _parse_list(row.get("options"))
        content = _as_text(row.get("question"))
        if options:
            content += "\nOptions:\n" + "\n".join(
                f"({chr(ord('A') + idx)}) {option}" for idx, option in enumerate(options)
            )
        prompt_text, ordered = _replace_numbered_images(content, image_map, sample_id=_as_text(row.get("id")))
        truth, score_type = _answer_and_score_type("mmmu", row.get("answer"), row.get("question_type"))
        prompt_text += "\nGive the final answer as the option letter."
        sample_id = _as_text(row.get("id"))
        rows.append({
            "id": f"mmmu:{sample_id}",
            "data_source": "mmmu",
            "prompt": [{"role": "user", "content": prompt_text}],
            "images": [{"bytes": image, "path": None} for image in ordered],
            "reward_model": {
                "ground_truth": truth, "benchmark": "mmmu", "score_type": score_type,
                "answers": [truth], "options": options, "precision": None,
            },
            "extra_info": {
                "index": -1, "sample_id": sample_id, "split": "validation",
                "metadata_json": json.dumps({"subfield": row.get("subfield"),
                                              "topic_difficulty": row.get("topic_difficulty"),
                                              "question_type": row.get("question_type")},
                                             ensure_ascii=False, sort_keys=True, default=str),
            },
        })
    return rows


LOADERS = [
    ("mathverse", _load_mathverse),
    ("mathvision", _load_mathvision),
    ("mathvista", _load_mathvista),
    ("wemath", _load_wemath),
    ("hallusionbench", _load_hallusionbench),
    ("chartqa", _load_chartqa),
    ("mmmu", _normalize_mmmu_rows),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_sources(download: bool) -> list[Path]:
    if download:
        mathvision_dir = STAGING / "mathvision-probe" / "data"
        hf_hub_download("MathLLMs/MathVision", "data/test-00000-of-00001-3532b8d3f1b4047a.parquet",
                        repo_type="dataset", local_dir=str(STAGING / "mathvision-probe"))
        hall_dir = STAGING / "hallusionbench-probe"
        for filename in ("data/image-00000-of-00001.parquet", "data/non_image-00000-of-00001.parquet"):
            hf_hub_download("lmms-lab/HallusionBench", filename, repo_type="dataset", local_dir=str(hall_dir))
        mmmu_dir = STAGING / "mmmu-probe" / "official"
        api = HfApi()
        names = sorted(name for name in api.list_repo_files("MMMU/MMMU", repo_type="dataset")
                       if name.count("/") == 1 and name.endswith("/validation-00000-of-00001.parquet"))
        if len(names) != 30:
            raise RuntimeError(f"Expected 30 official MMMU validation configs, found {len(names)}")
        for name in names:
            hf_hub_download("MMMU/MMMU", name, repo_type="dataset", local_dir=str(mmmu_dir))

    # These source files came with the previously downloaded raw dataset set.
    required = [
        STAGING / "mathverse-proxy" / "testmini.parquet",
        next((STAGING / "mathvision-probe" / "data").glob("test-*.parquet")),
        next((STAGING / "mathvista-probe").glob("testmini-*.parquet")),
        STAGING / "wemath-probe" / "We-Math.zip",
        STAGING / "hallusionbench-probe" / "data" / "image-00000-of-00001.parquet",
        STAGING / "hallusionbench-probe" / "data" / "non_image-00000-of-00001.parquet",
        STAGING / "chartqa-full-proxy" / "ChartQA Dataset.zip",
        *sorted((STAGING / "mmmu-probe" / "official").glob("*" + "/validation-*.parquet")),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))
    return required


def _write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    pq.write_table(table, path, compression="zstd", compression_level=4, use_dictionary=["data_source"])


def _image_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    return sum(bool(row["images"]) for row in rows), sum(len(row["images"]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true", help="Use only already downloaded source files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    source_files = _ensure_sources(download=not args.no_download)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Agent0-VL main-table, seven-benchmark normalized evaluation suite",
        "schema": "verl RLHFDataset Parquet: prompt/images/reward_model/data_source/extra_info",
        "source_repositories": {
            "mathverse": "AI4Math/MathVerse (existing testmini Parquet)",
            "mathvision": "MathLLMs/MathVision (official test Parquet, 3040 examples)",
            "mathvista": "AI4Math/MathVista (existing testmini Parquet)",
            "wemath": "We-Math/We-Math (existing official testmini archive)",
            "hallusionbench": "lmms-lab/HallusionBench (formatted official image + non_image splits)",
            "chartqa": "vis-nlp/ChartQA release (existing official test annotations/images archive)",
            "mmmu": "MMMU/MMMU official Hugging Face repository (30 validation subject configs)",
        },
        "benchmarks": {},
        "sources": [],
    }

    for benchmark, loader in LOADERS:
        rows = loader()
        expected = EXPECTED_ROWS[benchmark]
        if len(rows) != expected:
            raise RuntimeError(f"{benchmark}: expected {expected} rows for paper evaluation split; got {len(rows)}")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{benchmark}: duplicate normalized sample IDs")
        for idx, row in enumerate(rows):
            row["extra_info"]["index"] = idx
        image_rows, image_count = _image_counts(rows)
        output = args.output_dir / f"{benchmark}.parquet"
        _write_dataset(output, rows)
        table = pq.ParquetFile(output)
        if table.metadata.num_rows != expected:
            raise RuntimeError(f"{output}: wrote {table.metadata.num_rows} rows, expected {expected}")
        split_counts = Counter(row["extra_info"]["split"] for row in rows)
        manifest["benchmarks"][benchmark] = {
            "file": output.name,
            "rows": len(rows),
            "split_counts": dict(split_counts),
            "rows_with_images": image_rows,
            "total_images": image_count,
            "sha256": _sha256(output),
        }
        print(f"{benchmark}: rows={len(rows)}, rows_with_images={image_rows}, images={image_count}, file={output}", flush=True)
        all_rows.extend(rows)

    combined_path = args.output_dir / "paper_main_7.parquet"
    _write_dataset(combined_path, all_rows)
    manifest["total_rows"] = len(all_rows)
    manifest["combined_file"] = combined_path.name
    manifest["combined_sha256"] = _sha256(combined_path)
    manifest["sources"] = [{"path": str(path.relative_to(PROJECT_ROOT)), "bytes": path.stat().st_size,
                            "sha256": _sha256(path)} for path in source_files]
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"Combined evaluation set: {len(all_rows)} rows at {combined_path}", flush=True)


if __name__ == "__main__":
    main()
