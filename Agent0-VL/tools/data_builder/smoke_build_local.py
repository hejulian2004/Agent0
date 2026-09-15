"""Build a small local-Qwen smoke dataset from the downloaded paper sources.

This is intentionally a bounded, single-turn smoke builder.  It is useful for
checking the raw-data -> image -> Teacher -> canonical SFT record path before
the full tool-in-the-loop trajectory builder is enabled.  It does not mutate
``data/raw`` and it does not claim to reproduce the paper's undisclosed 200k /
40k sampling recipe.

The script produces exactly ``--samples-per-dataset`` valid solver-positive
records per formal paper source when the requested number of source samples and
Teacher generations succeed.  Invalid generations go to a manual-review JSONL
file rather than being silently turned into training targets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq

from tools.data_builder.backends import (
    GenerationChunk,
    TeacherBackendError,
    TeacherConfig,
    create_teacher_backend,
)
from tools.data_builder.schema import (
    ImageAsset,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    SCORER_VERSION,
    TaskRecord,
    canonical_json,
    sha256_text,
)


def _load_canonical_prompt_module() -> Any:
    """Load the prompt source without importing the heavyweight ``verl`` root."""

    module_path = Path(__file__).resolve().parents[2] / "verl" / "prompts" / "agent0_templates.py"
    spec = importlib.util.spec_from_file_location("agent0vl_canonical_prompt", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load canonical prompt module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PROMPTS = _load_canonical_prompt_module()
parse_final_answer = _PROMPTS.parse_final_answer
parse_solver_turn = _PROMPTS.parse_solver_turn
render_solver_request = _PROMPTS.render_solver_request
render_system_prompt = _PROMPTS.render_system_prompt


FORMAL_DATASETS = (
    "geometry3k",
    "geoqa",
    "mulberry",
    "mm_eureka",
    "retool",
    "mathverse",
    "mathvista",
    "wemath",
    "arxivqa",
    "chartqa",
    "thinklite",
)


DATASET_CAPABILITIES: dict[str, list[str]] = {
    "geometry3k": ["tool_use", "image_manipulation"],
    "geoqa": ["tool_use", "image_manipulation"],
    "mulberry": ["tool_use", "image_manipulation"],
    "mm_eureka": ["math_code"],
    "retool": ["tool_use", "math_code"],
    "mathverse": ["math_code"],
    "mathvista": ["math_code"],
    "wemath": ["math_code"],
    "arxivqa": ["tool_use"],
    "chartqa": ["tool_use", "image_manipulation"],
    "thinklite": ["math_code"],
}


@dataclass
class SourceSample:
    dataset: str
    question: str
    ground_truth: str
    original_id: str
    official_split: str
    source_revision: str | None
    source_license: str | None
    source_license_status: str
    answer_type: str = "exact"
    accepted_answers: list[str] = field(default_factory=list)
    image_bytes: bytes | None = None
    image_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PersistedSample:
    sample: SourceSample
    image_path: Path | None
    image_hash: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_raw_report(repo_root: Path) -> dict[str, Any]:
    report_path = repo_root / "data" / "raw" / "raw_download_report.json"
    if not report_path.exists():
        return {"datasets": []}
    with report_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {"datasets": []}


def _report_entries(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in report.get("datasets", []):
        if isinstance(entry, Mapping) and entry.get("name"):
            result[str(entry["name"])] = dict(entry)
    return result


def _raw_dir(
    repo_root: Path,
    entries: Mapping[str, Mapping[str, Any]],
    dataset: str,
    fallback_folder: str,
) -> Path:
    entry = entries.get(dataset, {})
    value = entry.get("local_path")
    if value:
        path = Path(str(value))
        if not path.is_absolute():
            path = repo_root / path
        return path
    return repo_root / "data" / "raw" / ".staging" / fallback_folder


def _source_metadata(
    entries: Mapping[str, Mapping[str, Any]], dataset: str
) -> tuple[str | None, str | None, str]:
    entry = entries.get(dataset, {})
    revision = entry.get("resolved_revision")
    license_value = entry.get("license")
    license_status = str(entry.get("license_status") or "unknown")
    return (
        str(revision) if revision is not None else None,
        str(license_value) if license_value is not None else None,
        license_status,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _image_suffix(value: bytes) -> str | None:
    if value.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if value.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if value.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if value.startswith(b"RIFF") and value[8:12] == b"WEBP":
        return "webp"
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value).replace("\x00", "").strip()


def _trim_question(value: Any, limit: int = 3600) -> str:
    text = _clean_text(value)
    text = text.replace("<image>", "").replace("<|image|>", "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[QUESTION_TRUNCATED_FOR_SMOKE]"


def _extract_question(value: Any) -> str:
    text = _clean_text(value).replace("<image>", "").strip()
    # Keep the question after common dataset instruction prefixes.
    matches = list(re.finditer(r"(?:\*\*user question:\*\*|Question:)\s*", text, re.I))
    if matches:
        text = text[matches[-1].end():]
    return _trim_question(text)


def _answer_from_text(value: Any) -> str:
    text = _clean_text(value)
    canonical = parse_final_answer(text, allow_legacy_boxed=True)
    if canonical:
        return canonical
    match = re.search(r"(?:The final answer is|final answer is|answer is)\s*:?\s*(.+)", text, re.I)
    if match:
        return match.group(1).splitlines()[0].strip().strip(".$")
    return _trim_question(text, limit=400)


def _choices_text(choices: Any) -> str:
    if choices is None:
        return ""
    if isinstance(choices, Mapping):
        values = [f"{key}: {value}" for key, value in choices.items()]
    elif isinstance(choices, (list, tuple)):
        values = [f"{index + 1}. {_clean_text(value)}" for index, value in enumerate(choices)]
    else:
        return _clean_text(choices)
    return "\n".join(values)


def _with_choices(question: str, choices: Any) -> str:
    rendered = _choices_text(choices)
    if not rendered or rendered in question:
        return _trim_question(question)
    return _trim_question(f"{question}\n\nChoices:\n{rendered}")


def _choice_answer(label: Any, choices: Sequence[Any] | None = None) -> tuple[str, list[str]]:
    if isinstance(label, int):
        if choices is not None and 0 <= label < len(choices):
            value = _clean_text(choices[label])
            letter = chr(ord("A") + label)
            return letter, [letter, value]
        return str(label), [str(label)]
    text = _clean_text(label)
    if text.isdigit() and choices is not None:
        index = int(text)
        if 0 <= index < len(choices):
            letter = chr(ord("A") + index)
            return letter, [letter, _clean_text(choices[index])]
    return text, [text] if text else []


class ZipLookup:
    """Small suffix-aware lookup over a local zip central directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.names = [name for name in self.archive.namelist() if not name.endswith("/")]
        self.name_set = set(self.names)

    def find(self, reference: str) -> str | None:
        normalized = reference.replace("\\", "/").lstrip("/")
        candidates = [normalized]
        if normalized.startswith("./"):
            candidates.append(normalized[2:])
        for candidate in candidates:
            if candidate in self.name_set:
                return candidate
        suffix = "/" + candidates[-1]
        for name in self.names:
            if name.endswith(suffix) or name.endswith(candidates[-1]):
                return name
        return None

    def read(self, reference: str) -> tuple[bytes, str] | None:
        member = self.find(reference)
        if member is None:
            return None
        return self.archive.read(member), member

    def close(self) -> None:
        self.archive.close()

    def __enter__(self) -> "ZipLookup":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Stream a JSON array without loading a 486 MB Mulberry file in memory."""

    decoder = json.JSONDecoder()
    buffer = ""
    eof = False
    started = False
    first_item = True

    with path.open("r", encoding="utf-8") as handle:
        while True:
            while not buffer and not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            if not buffer and eof:
                return

            if not started:
                buffer = buffer.lstrip()
                if not buffer and not eof:
                    continue
                if not buffer or buffer[0] != "[":
                    raise ValueError(f"expected a JSON array in {path}")
                buffer = buffer[1:]
                started = True

            buffer = buffer.lstrip()
            if not buffer and not eof:
                continue
            if not buffer:
                raise ValueError(f"unterminated JSON array in {path}")

            if not first_item:
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    buffer = buffer.lstrip()
                    if not buffer and not eof:
                        continue
                elif buffer[0] == "]":
                    return
                else:
                    raise ValueError(f"expected a comma in {path}")
            if buffer and buffer[0] == "]":
                return

            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    chunk = handle.read(chunk_size)
                    if chunk:
                        buffer += chunk
                    else:
                        eof = True
            buffer = buffer[end:]
            first_item = False
            yield value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def _iter_parquet_rows(path: Path, columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=10, columns=list(columns)):
        yield from batch.to_pylist()


def _make_sample(
    *,
    dataset: str,
    question: str,
    ground_truth: str,
    original_id: Any,
    official_split: str,
    source_revision: str | None,
    source_license: str | None,
    source_license_status: str,
    answer_type: str = "exact",
    accepted_answers: Sequence[str] | None = None,
    image_bytes: bytes | None = None,
    image_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SourceSample:
    return SourceSample(
        dataset=dataset,
        question=_trim_question(question),
        ground_truth=_clean_text(ground_truth),
        original_id=_clean_text(original_id),
        official_split=official_split,
        source_revision=source_revision,
        source_license=source_license,
        source_license_status=source_license_status,
        answer_type=answer_type,
        accepted_answers=list(accepted_answers or []),
        image_bytes=image_bytes,
        image_name=image_name,
        metadata=dict(metadata or {}),
    )


def _metadata_for(
    entries: Mapping[str, Mapping[str, Any]], dataset: str
) -> tuple[str | None, str | None, str]:
    return _source_metadata(entries, dataset)


def load_geometry3k(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "geometry3k")
    raw = _raw_dir(repo_root, entries, "geometry3k", "geometry3k-probe")
    archive_path = raw / "raw" / "train.zip"
    result: list[SourceSample] = []
    with zipfile.ZipFile(archive_path) as archive:
        data_names = sorted(name for name in archive.namelist() if name.endswith("/data.json"))
        for name in data_names:
            data = json.loads(archive.read(name))
            choices = data.get("choices") or data.get("compact_choices")
            question = _with_choices(data.get("problem_text") or data.get("compact_text"), choices)
            answer = data.get("answer", "")
            answer_text, accepted = _choice_answer(answer, choices if isinstance(choices, list) else None)
            folder = name.rsplit("/", 1)[0]
            image_member = f"{folder}/img_diagram.png"
            image_bytes = archive.read(image_member) if image_member in archive.namelist() else None
            result.append(_make_sample(
                dataset="geometry3k", question=question, ground_truth=answer_text,
                original_id=data.get("id"), official_split="train", source_revision=revision,
                source_license=license_value, source_license_status=license_status,
                answer_type="choice", accepted_answers=accepted, image_bytes=image_bytes,
                image_name=image_member, metadata={"source_file": name},
            ))
            if len(result) >= n:
                break
    return result


def load_geoqa(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "geoqa")
    raw = _raw_dir(repo_root, entries, "geoqa", "geoqa-official-proxy")
    result: list[SourceSample] = []
    with zipfile.ZipFile(raw / "data.zip") as archive:
        names = sorted(
            (name for name in archive.namelist() if name.startswith("GeoQA3/json/") and name.endswith(".json")),
            key=lambda name: int(Path(name).stem) if Path(name).stem.isdigit() else name,
        )
        member_names = set(archive.namelist())
        for name in names:
            data = json.loads(archive.read(name))
            choices = data.get("choices") or []
            answer, accepted = _choice_answer(data.get("label"), choices)
            image_member = f"GeoQA3/image/{Path(name).stem}.png"
            image_bytes = archive.read(image_member) if image_member in member_names else None
            result.append(_make_sample(
                dataset="geoqa", question=_with_choices(data.get("subject", ""), choices),
                ground_truth=answer, original_id=data.get("id", Path(name).stem),
                official_split="train", source_revision=revision, source_license=license_value,
                source_license_status=license_status, answer_type="choice", accepted_answers=accepted,
                image_bytes=image_bytes, image_name=image_member, metadata={"source_file": name},
            ))
            if len(result) >= n:
                break
    return result


def _mulberry_row_question(row: Mapping[str, Any]) -> str:
    messages = row.get("messages") or []
    users = [message.get("content", "") for message in messages if isinstance(message, Mapping) and message.get("role") == "user"]
    return _extract_question(users[-1] if users else "")


def _mulberry_row_answer(row: Mapping[str, Any]) -> str:
    messages = row.get("messages") or []
    assistants = [message.get("content", "") for message in messages if isinstance(message, Mapping) and message.get("role") == "assistant"]
    return _answer_from_text(assistants[-1] if assistants else "")


def load_mulberry(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "mulberry")
    raw = _raw_dir(repo_root, entries, "mulberry", "mulberry-proxy")
    json_path = raw / "mulberry_sft.json"
    candidates: list[tuple[int, Mapping[str, Any], str]] = []
    for index, row in enumerate(_iter_json_array(json_path)):
        if not isinstance(row, Mapping) or not row.get("images"):
            continue
        reference = _clean_text(row.get("images")).lstrip("/")
        candidates.append((index, row, "mulberry_images/" + reference))
        if len(candidates) >= max(300, n * 30):
            break

    targets = {target for _, _, target in candidates}
    found: dict[str, bytes] = {}
    with tarfile.open(raw / "mulberry_images.tar", "r:*") as archive:
        for member_index, member in enumerate(archive):
            if member.name in targets and member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    found[member.name] = handle.read()
            if len(found) >= n or member_index >= 25000:
                break

    result: list[SourceSample] = []
    for index, row, target in candidates:
        if target not in found:
            continue
        result.append(_make_sample(
            dataset="mulberry", question=_mulberry_row_question(row),
            ground_truth=_mulberry_row_answer(row), original_id=f"mulberry-{index}",
            official_split="train", source_revision=revision, source_license=license_value,
            source_license_status=license_status, answer_type="exact",
            accepted_answers=[_mulberry_row_answer(row)], image_bytes=found[target],
            image_name=target, metadata={"source_index": index},
        ))
        if len(result) >= n:
            break
    return result


def _conversation_user_text(row: Mapping[str, Any]) -> str:
    conversations = row.get("conversations") or []
    users = [item.get("content", "") for item in conversations if isinstance(item, Mapping) and item.get("role") == "user"]
    return _extract_question(users[-1] if users else "")


def load_mm_eureka(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "mm_eureka")
    raw = _raw_dir(repo_root, entries, "mm_eureka", "mmeureka-proxy")
    result: list[SourceSample] = []
    with ZipLookup(raw / "MMPR.zip") as mmpr, ZipLookup(raw / "K12.zip") as k12:
        for row_index, row in enumerate(_iter_jsonl(raw / "dataset.jsonl")):
            urls = row.get("image_urls") or []
            if isinstance(urls, str):
                urls = [urls]
            image_bytes = None
            image_name = None
            for reference in urls:
                ref = _clean_text(reference)
                loaded = mmpr.read(ref) or k12.read(ref)
                if loaded is not None:
                    image_bytes, image_name = loaded
                    break
            if image_bytes is None:
                continue
            answer = _clean_text(row.get("answer"))
            result.append(_make_sample(
                dataset="mm_eureka", question=_conversation_user_text(row), ground_truth=answer,
                original_id=row.get("id", f"mm-eureka-{row_index}"), official_split="train",
                source_revision=revision, source_license=license_value, source_license_status=license_status,
                answer_type="math", accepted_answers=[answer] if answer else [], image_bytes=image_bytes,
                image_name=image_name, metadata={"source_index": row_index, "image_url": urls[0] if urls else None},
            ))
            if len(result) >= n:
                break
    return result


def load_retool(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "retool")
    raw = _raw_dir(repo_root, entries, "retool", "retool-proxy")
    result: list[SourceSample] = []
    columns = ["messages", "tools"]
    for row_index, row in enumerate(_iter_parquet_rows(raw / "train_2000.parquet", columns)):
        messages = row.get("messages") or []
        user_values = [item.get("content", "") for item in messages if isinstance(item, Mapping) and item.get("role") == "user"]
        assistant_values = [item.get("content", "") for item in messages if isinstance(item, Mapping) and item.get("role") == "assistant"]
        question = _extract_question(user_values[-1] if user_values else "")
        answer = _answer_from_text(assistant_values[-1] if assistant_values else "")
        result.append(_make_sample(
            dataset="retool", question=question, ground_truth=answer,
            original_id=f"retool-{row_index}", official_split="train", source_revision=revision,
            source_license=license_value, source_license_status=license_status, answer_type="math",
            accepted_answers=[answer] if answer else [], metadata={"source_index": row_index},
        ))
        if len(result) >= n:
            break
    return result


def load_mathverse(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "mathverse")
    raw = _raw_dir(repo_root, entries, "mathverse", "mathverse-proxy")
    result: list[SourceSample] = []
    columns = ["sample_index", "problem_index", "question", "image", "answer", "metadata"]
    for row in _iter_parquet_rows(raw / "testmini.parquet", columns):
        image = row.get("image")
        image_bytes = image.get("bytes") if isinstance(image, Mapping) else image if isinstance(image, (bytes, bytearray)) else None
        question = _with_choices(row.get("question", ""), None)
        answer = _clean_text(row.get("answer"))
        result.append(_make_sample(
            dataset="mathverse", question=question, ground_truth=answer,
            original_id=f"{row.get('problem_index', 'unknown')}-{row.get('sample_index', len(result))}",
            official_split="testmini", source_revision=revision, source_license=license_value,
            source_license_status=license_status, answer_type="choice" if re.fullmatch(r"[A-E]", answer, re.I) else "exact",
            accepted_answers=[answer] if answer else [], image_bytes=bytes(image_bytes) if image_bytes is not None else None,
            image_name=f"mathverse-{row.get('sample_index', len(result))}", metadata={"metadata": row.get("metadata")},
        ))
        if len(result) >= n:
            break
    return result


def load_mathvista(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "mathvista")
    raw = _raw_dir(repo_root, entries, "mathvista", "mathvista-probe")
    result: list[SourceSample] = []
    image_zip = raw / "images.zip"
    columns = ["pid", "question", "image", "choices", "answer", "answer_type", "metadata"]
    with ZipLookup(image_zip) as images:
        for row in _iter_parquet_rows(raw / "testmini-00000-of-00001-725687bf7a18d64b.parquet", columns):
            image_reference = _clean_text(row.get("image"))
            loaded = images.read(image_reference) if image_reference else None
            image_bytes, image_name = (loaded if loaded is not None else (None, image_reference))
            answer = _clean_text(row.get("answer"))
            result.append(_make_sample(
                dataset="mathvista", question=_with_choices(row.get("question", ""), row.get("choices")),
                ground_truth=answer, original_id=row.get("pid"), official_split="testmini",
                source_revision=revision, source_license=license_value, source_license_status=license_status,
                answer_type=_clean_text(row.get("answer_type")) or "exact", accepted_answers=[answer] if answer else [],
                image_bytes=image_bytes, image_name=image_name, metadata={"metadata": row.get("metadata")},
            ))
            if len(result) >= n:
                break
    return result


def load_wemath(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "wemath")
    raw = _raw_dir(repo_root, entries, "wemath", "wemath-probe")
    result: list[SourceSample] = []
    with ZipLookup(raw / "We-Math.zip") as archive:
        payload = json.loads(archive.archive.read("We-Math/testmini.json"))
        for row in payload:
            reference = _clean_text(row.get("image_path"))
            loaded = archive.read("We-Math/data/" + reference) if reference else None
            image_bytes, image_name = (loaded if loaded is not None else (None, reference))
            answer = _clean_text(row.get("answer"))
            result.append(_make_sample(
                dataset="wemath", question=_with_choices(row.get("question", ""), row.get("option")),
                ground_truth=answer, original_id=row.get("ID"), official_split="testmini",
                source_revision=revision, source_license=license_value, source_license_status=license_status,
                answer_type="choice", accepted_answers=[answer] if answer else [], image_bytes=image_bytes,
                image_name=image_name, metadata={"knowledge_concept": row.get("knowledge concept")},
            ))
            if len(result) >= n:
                break
    return result


def load_arxivqa(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "arxivqa")
    raw = _raw_dir(repo_root, entries, "arxivqa", "arxivqa-probe")
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(raw / "arxivqa.jsonl"):
        rows.append(row)
        if len(rows) >= max(1000, n * 50):
            break
    targets = {_clean_text(row.get("image")) for row in rows if row.get("image")}
    found: dict[str, bytes] = {}
    with tarfile.open(raw / "images.tgz", "r:*") as archive:
        for member_index, member in enumerate(archive):
            if member.name in targets and member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    found[member.name] = handle.read()
            if len(found) >= n or member_index >= 25000:
                break
    result: list[SourceSample] = []
    for row_index, row in enumerate(rows):
        reference = _clean_text(row.get("image"))
        if reference not in found:
            continue
        label = _clean_text(row.get("label"))
        choices = row.get("options") or []
        result.append(_make_sample(
            dataset="arxivqa", question=_with_choices(row.get("question", ""), choices),
            ground_truth=label, original_id=row.get("id", f"arxivqa-{row_index}"), official_split="train",
            source_revision=revision, source_license=license_value, source_license_status=license_status,
            answer_type="choice", accepted_answers=[label] if label else [], image_bytes=found[reference],
            image_name=reference, metadata={"source_index": row_index, "rationale_available": bool(row.get("rationale"))},
        ))
        if len(result) >= n:
            break
    return result


def load_chartqa(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "chartqa")
    raw = _raw_dir(repo_root, entries, "chartqa", "chartqa-full-proxy")
    result: list[SourceSample] = []
    with ZipLookup(raw / "ChartQA Dataset.zip") as archive:
        payload = json.loads(archive.archive.read("ChartQA Dataset/train/train_human.json"))
        for index, row in enumerate(payload):
            reference = _clean_text(row.get("imgname"))
            loaded = archive.read("ChartQA Dataset/train/png/" + reference)
            if loaded is None:
                continue
            image_bytes, image_name = loaded
            answer = _clean_text(row.get("label"))
            result.append(_make_sample(
                dataset="chartqa", question=row.get("query", ""), ground_truth=answer,
                original_id=reference.rsplit(".", 1)[0], official_split="train", source_revision=revision,
                source_license=license_value, source_license_status=license_status, answer_type="exact",
                accepted_answers=[answer] if answer else [], image_bytes=image_bytes, image_name=image_name,
                metadata={"source_index": index},
            ))
            if len(result) >= n:
                break
    return result


def load_thinklite(repo_root: Path, entries: Mapping[str, Mapping[str, Any]], n: int) -> list[SourceSample]:
    revision, license_value, license_status = _metadata_for(entries, "thinklite")
    raw = _raw_dir(repo_root, entries, "thinklite", "thinklite-proxy")
    result: list[SourceSample] = []
    columns = ["image", "problem", "answer", "id", "choices", "ground_truth"]
    for row in _iter_parquet_rows(raw / "ThinkLite-VL-70k.parquet", columns):
        image = row.get("image")
        image_bytes = image.get("bytes") if isinstance(image, Mapping) else image if isinstance(image, (bytes, bytearray)) else None
        answer = _clean_text(row.get("ground_truth") or row.get("answer"))
        choices = row.get("choices")
        result.append(_make_sample(
            dataset="thinklite", question=_with_choices(row.get("problem", ""), choices), ground_truth=answer,
            original_id=row.get("id"), official_split="train", source_revision=revision,
            source_license=license_value, source_license_status=license_status,
            answer_type="choice" if choices else "exact", accepted_answers=[answer] if answer else [],
            image_bytes=bytes(image_bytes) if image_bytes is not None else None,
            image_name=f"thinklite-{row.get('id', len(result))}", metadata={},
        ))
        if len(result) >= n:
            break
    return result


LOADERS = {
    "geometry3k": load_geometry3k,
    "geoqa": load_geoqa,
    "mulberry": load_mulberry,
    "mm_eureka": load_mm_eureka,
    "retool": load_retool,
    "mathverse": load_mathverse,
    "mathvista": load_mathvista,
    "wemath": load_wemath,
    "arxivqa": load_arxivqa,
    "chartqa": load_chartqa,
    "thinklite": load_thinklite,
}


def _stage_for(dataset: str) -> str:
    capabilities = DATASET_CAPABILITIES[dataset]
    return "stage2" if "math_code" in capabilities else "stage1"


def _persist_image(sample: SourceSample, assets_root: Path) -> PersistedSample:
    if not sample.image_bytes:
        return PersistedSample(sample=sample, image_path=None, image_hash=None)
    image_hash = _sha256_bytes(sample.image_bytes)
    suffix = _image_suffix(sample.image_bytes)
    if suffix is None:
        return PersistedSample(sample=sample, image_path=None, image_hash=image_hash)
    dataset_dir = assets_root / sample.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", sample.original_id or "sample")[:80]
    output_path = dataset_dir / f"{safe_id}-{image_hash[:12]}.{suffix}"
    if not output_path.exists():
        output_path.write_bytes(sample.image_bytes)
    return PersistedSample(sample=sample, image_path=output_path, image_hash=image_hash)


def _hash_question(question: str) -> str:
    return sha256_text(" ".join(question.split()))


def _teacher_prompt(sample: SourceSample, retry_note: str = "") -> str:
    image_note = "A visual input is attached; inspect it before answering." if sample.image_bytes else "There is no image; solve from the text."
    instructions = (
        "This is a bounded smoke-data generation request. Produce exactly one FINAL_TURN. "
        "Do not emit Python, fenced code, JSON, or a tool call. Use at most two short "
        "reasoning sentences and do not add an image description.\n\n"
        "Required final format (both lines, each exactly once):\n"
        "CONFIDENCE: <number between 0 and 1>\n"
        "FINAL_ANSWER: <single-line answer>\n"
        "Return the two required lines at the end and do not output any other protocol.\n\n"
        f"{image_note}"
    )
    if retry_note:
        instructions += f"\n\n{retry_note}"
    return render_solver_request(
        f"{instructions}\n\nQuestion:\n{sample.question}",
        image_context="one attached image" if sample.image_bytes else "",
    )


def _task_record(sample: SourceSample, image_path: Path | None, image_hash: str | None) -> TaskRecord:
    image_assets: list[ImageAsset] = []
    if image_path is not None and image_hash is not None:
        image_assets.append(ImageAsset(
            asset_id=f"{sample.dataset}-{sample.original_id}-{image_hash[:12]}",
            sha256=image_hash,
            path=str(image_path),
        ))
    return TaskRecord(
        task_id=f"{sample.dataset}:{sample.original_id}",
        source=sample.dataset,
        question=sample.question,
        ground_truth=sample.ground_truth,
        answer_type=sample.answer_type if sample.answer_type in {"math", "exact", "choice", "list"} else "exact",
        source_revision=sample.source_revision,
        license=sample.source_license,
        original_id=sample.original_id,
        official_split=sample.official_split,
        accepted_answers=sample.accepted_answers,
        capability_labels=DATASET_CAPABILITIES[sample.dataset],
        images=image_assets,
        hashes={"question_hash": _hash_question(sample.question), **({"image_hash": image_hash} if image_hash else {})},
    )


def _sample_record_key(sample: SourceSample) -> str:
    """Return a stable per-source key; ChartQA has multiple questions/image."""

    source_index = sample.metadata.get("source_index")
    if source_index is not None:
        return str(source_index)
    return sample.original_id


def _record(
    persisted: PersistedSample,
    chunk: GenerationChunk,
    attempt: int,
    parsed: Any,
    output_root: Path,
) -> dict[str, Any]:
    sample = persisted.sample
    user_content = _teacher_prompt(sample)
    if persisted.image_path is not None:
        user_content = "<image>\n" + user_content
    task = _task_record(sample, persisted.image_path, persisted.image_hash)
    record_id = f"{sample.dataset}-{_sample_record_key(sample)}-solver-positive"
    return {
        "record_id": record_id,
        "record_type": "solver_positive",
        "source_trajectory_id": f"{sample.dataset}:{sample.original_id}:local-qwen-smoke",
        "source_step_index": 1,
        "stage": _stage_for(sample.dataset),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": chunk.text.strip()},
        ],
        "images": [
            str(persisted.image_path.relative_to(output_root).as_posix())
        ] if persisted.image_path is not None else [],
        "loss_target": "last_assistant_only",
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "scorer_version": SCORER_VERSION,
        "metadata": {
            "task": {
                "task_id": task.task_id,
                "source": task.source,
                "original_id": task.original_id,
                "official_split": task.official_split,
                "ground_truth": task.ground_truth,
                "answer_type": task.answer_type,
                "accepted_answers": task.accepted_answers,
                "capability_labels": task.capability_labels,
                "source_revision": task.source_revision,
                "license": task.license,
                "license_status": sample.source_license_status,
                "hashes": task.hashes,
                "source_metadata": sample.metadata,
            },
            "teacher": {
                "backend": chunk.backend,
                "model": chunk.model,
                "request_id": chunk.request_id,
                "finish_reason": chunk.finish_reason,
                "usage": chunk.usage,
                "attempt": attempt,
            },
            "parsed_solver_turn": {
                "turn_type": parsed.turn_type.value,
                "confidence": parsed.confidence,
                "final_answer": parsed.final_answer,
                "errors": parsed.errors,
            },
            "smoke_mode": "single_turn_final_only",
            "output_root": str(output_root),
        },
    }


def _manual_review_entry(
    persisted: PersistedSample,
    reason: str,
    attempt: int,
    raw_text: str = "",
) -> dict[str, Any]:
    sample = persisted.sample
    return {
        "record_id": f"{sample.dataset}-{_sample_record_key(sample)}-solver-positive",
        "dataset": sample.dataset,
        "original_id": sample.original_id,
        "attempt": attempt,
        "reason": reason,
        "raw_text": raw_text,
        "question_hash": _hash_question(sample.question),
        "image_hash": persisted.image_hash,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output).resolve()
    assets_root = output_root / "assets"
    output_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)

    report = _load_raw_report(repo_root)
    entries = _report_entries(report)
    config = TeacherConfig(
        backend="openai_compatible",
        base_url=args.base_url,
        model=args.model,
        allow_anonymous=args.allow_anonymous,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.backend_retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    backend = create_teacher_backend(config)
    rng = random.Random(args.seed)
    del rng  # Selection is source-order stable; seed is still recorded for provenance.

    jsonl_path = output_root / "sft_records.jsonl"
    review_path = output_root / "manual_review_queue.jsonl"
    selection_path = output_root / "selection_manifest.jsonl"
    summary: dict[str, Any] = {
        "build_mode": "local_qwen_single_turn_smoke",
        "paper_scope": "auditable reconstruction from public source list; not the undisclosed paper dataset recipe",
        "formal_datasets": list(FORMAL_DATASETS),
        "requested_samples_per_dataset": args.samples_per_dataset,
        "target_sft_supervision_records": len(FORMAL_DATASETS) * args.samples_per_dataset,
        "seed": args.seed,
        "teacher_config": config.public_dict(),
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "scorer_version": SCORER_VERSION,
        "datasets": {},
        "source_trajectory_count": 0,
        "sft_supervision_record_count": 0,
        "valid_record_count": 0,
        "manual_review_count": 0,
        "started_at_epoch": time.time(),
        "notes": [
            "This smoke path asks for one final Solver turn and does not execute Python or verifier/repair turns.",
            "MathVerse, MathVista, and We-Math use local testmini rows because the downloaded public release is eval-only; these records are debug-only and must not enter formal training.",
            "The raw dataset license statuses are preserved; manual_review_required sources are not formal-training eligible.",
        ],
    }

    with jsonl_path.open("w", encoding="utf-8") as records_handle, review_path.open("w", encoding="utf-8") as review_handle, selection_path.open("w", encoding="utf-8") as selection_handle:
        try:
            for dataset in FORMAL_DATASETS:
                dataset_info: dict[str, Any] = {
                    "requested": args.samples_per_dataset,
                    "selected": 0,
                    "generated": 0,
                    "valid": 0,
                    "manual_review": 0,
                    "stage": _stage_for(dataset),
                    "capability_labels": DATASET_CAPABILITIES[dataset],
                    "errors": [],
                }
                try:
                    samples = LOADERS[dataset](repo_root, entries, args.samples_per_dataset)
                except Exception as exc:  # keep other sources diagnosable
                    dataset_info["errors"].append(f"loader: {type(exc).__name__}: {exc}")
                    summary["datasets"][dataset] = dataset_info
                    continue

                persisted_samples = [_persist_image(sample, assets_root) for sample in samples]
                dataset_info["selected"] = len(persisted_samples)
                for persisted in persisted_samples:
                    selection_handle.write(json.dumps({
                        "dataset": dataset,
                        "original_id": persisted.sample.original_id,
                        "official_split": persisted.sample.official_split,
                        "question_hash": _hash_question(persisted.sample.question),
                        "image_hash": persisted.image_hash,
                        "image_path": str(persisted.image_path.relative_to(output_root).as_posix()) if persisted.image_path else None,
                        "source_revision": persisted.sample.source_revision,
                        "license_status": persisted.sample.source_license_status,
                    }, ensure_ascii=False) + "\n")
                    summary["source_trajectory_count"] += 1

                    retry_note = ""
                    last_raw = ""
                    valid_record: dict[str, Any] | None = None
                    for attempt in range(1, args.max_attempts + 1):
                        dataset_info["generated"] += 1
                        try:
                            context = [
                                {"role": "system", "content": render_system_prompt()},
                                {"role": "user", "content": _teacher_prompt(persisted.sample, retry_note)},
                            ]
                            image_values = [str(persisted.image_path)] if persisted.image_path else []
                            chunk = backend.generate_next(context, role="solver", images=image_values)
                            last_raw = chunk.text
                            parsed = parse_solver_turn(chunk.text, allow_legacy_boxed=False)
                            if parsed.turn_type.value != "final":
                                retry_note = (
                                    "The previous output was invalid because it was not a canonical FINAL_TURN. "
                                    "Return no code and exactly one CONFIDENCE line plus one FINAL_ANSWER line."
                                )
                                continue
                            if parse_final_answer(chunk.text, allow_legacy_boxed=False) is None:
                                retry_note = "FINAL_ANSWER was missing or ambiguous. Return exactly one non-empty FINAL_ANSWER line."
                                continue
                            valid_record = _record(persisted, chunk, attempt, parsed, output_root)
                            break
                        except (TeacherBackendError, OSError, ValueError) as exc:
                            dataset_info["errors"].append(
                                f"{persisted.sample.original_id} attempt {attempt}: {type(exc).__name__}: {exc}"
                            )
                            retry_note = "The previous request failed. Return only a short canonical FINAL_TURN."

                    if valid_record is None:
                        dataset_info["manual_review"] += 1
                        summary["manual_review_count"] += 1
                        review_handle.write(json.dumps(
                            _manual_review_entry(persisted, "no_valid_final_turn", args.max_attempts, last_raw),
                            ensure_ascii=False,
                        ) + "\n")
                        continue

                    # Enforce the Phase 0 SFT invariant at the final write boundary.
                    messages = valid_record["messages"]
                    if sum(message.get("role") == "assistant" for message in messages) != 1 or messages[-1].get("role") != "assistant":
                        raise AssertionError("smoke exporter produced a non-single-assistant record")
                    records_handle.write(json.dumps(valid_record, ensure_ascii=False) + "\n")
                    dataset_info["valid"] += 1
                    summary["valid_record_count"] += 1
                    summary["sft_supervision_record_count"] += 1

                summary["datasets"][dataset] = dataset_info
        finally:
            backend.close()

    summary["finished_at_epoch"] = time.time()
    summary["formal_target_met"] = summary["valid_record_count"] == summary["target_sft_supervision_records"]
    summary["dataset_target_status"] = {
        name: info.get("valid", 0) == args.samples_per_dataset
        for name, info in summary["datasets"].items()
    }
    report_path = output_root / "build_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = _repo_root()
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_default / "data" / "smoke" / "local_qwen_27b_10_per_dataset",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="Qwen3.8-27B-NVFP4-Q5K-no-MTP")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--allow-anonymous", action="store_true", default=True)
    parser.add_argument("--samples-per-dataset", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--backend-retries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.samples_per_dataset <= 0:
        raise SystemExit("--samples-per-dataset must be positive")
    summary = build(args)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "valid_record_count": summary["valid_record_count"],
        "target_sft_supervision_records": summary["target_sft_supervision_records"],
        "formal_target_met": summary["formal_target_met"],
        "manual_review_count": summary["manual_review_count"],
        "datasets": summary["dataset_target_status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
