"""Small source adapters for the Agent0-VL SFT builder.

The public datasets used by the paper are available in several local shapes:
raw Geometry3K directories, Hugging Face exports, and verl-style ReTool rows.
This module normalizes those shapes without putting source metadata into the
teacher conversation.

The normalized record intentionally has only the fields needed by the
builder.  ``ground_truth`` remains an internal post-rollout check and is
never copied into a teacher request or an exported SFT row.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SOURCE_STAGES = {
    "geometry3k": 1,
    "geoqa": 1,
    "mulberry": 1,
    "retool": 2,
    "mmeureka": 2,
}

SOURCE_STAGE_OPTIONS = {
    "geometry3k": (1,),
    "geoqa": (1,),
    "mulberry": (1, 2),
    "retool": (1, 2),
    "mmeureka": (2,),
}

_SOURCE_ALIASES = {
    "geometry_3k": "geometry3k",
    "geometry3k_local": "geometry3k",
    "dapo": "retool",
    "dapo_math": "retool",
    "dapo_math_17k": "retool",
    "mm_eureka": "mmeureka",
}

_IMAGE_KEYS = (
    "images",
    "image",
    "image_path",
    "image_paths",
    "img",
    "img_path",
    "picture",
    "image_urls",
)

_QUESTION_KEYS = (
    "question",
    "problem",
    "problem_text",
    "annotat_text",
    "query",
    "instruction",
    "subject",
)


class SourceFormatError(ValueError):
    """Raised when a source row cannot be normalized safely."""


def canonical_source_name(source: str) -> str:
    """Return the supported canonical name for a source alias."""

    name = source.strip().lower()
    name = _SOURCE_ALIASES.get(name, name)
    if name not in SOURCE_STAGES:
        supported = ", ".join(sorted(SOURCE_STAGES))
        raise SourceFormatError(f"Unsupported source {source!r}; choose one of: {supported}")
    return name


def _as_text(value: Any) -> Optional[str]:
    """Convert common dataset text containers to one string."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "string"):
            if key in value:
                text = _as_text(value[key])
                if text:
                    return text
        return None
    if isinstance(value, (list, tuple)):
        parts = [_as_text(item) for item in value]
        joined = "".join(part for part in parts if part)
        return joined.strip() or None
    return str(value).strip() or None


def _unwrap_singleton(value: Any) -> Any:
    """Unwrap the singleton lists used by some parquet exports."""

    while True:
        # pandas returns nested parquet list columns as numpy arrays.  Keep
        # this adapter dependency-free by using the common ``tolist`` method.
        tolist = getattr(value, "tolist", None)
        if callable(tolist) and not isinstance(value, (str, bytes, dict)):
            converted = tolist()
            if converted is value:
                break
            value = converted
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        break
    return value


def _nested_value(row: Dict[str, Any], *path: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _read_json(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SourceFormatError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("data", "records", "examples", "items"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        else:
            split_rows = [
                value
                for value in data.values()
                if isinstance(value, list) and all(isinstance(item, dict) for item in value)
            ]
            rows = [item for split in split_rows for item in split] if split_rows else [data]
    else:
        raise SourceFormatError(f"Expected an object or list in {path}, got {type(data).__name__}")

    result: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SourceFormatError(f"Row {index} in {path} is not an object")
        result.append(row)
    return result


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceFormatError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SourceFormatError(f"Row {line_number} in {path} is not an object")
            result.append(row)
    return result


def _read_parquet(path: Path) -> List[Dict[str, Any]]:
    """Read parquet through the repository's existing pandas dependency."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on the local install
        raise SourceFormatError(
            f"Reading parquet requires pandas/pyarrow; cannot read {path}"
        ) from exc
    try:
        frame = pd.read_parquet(path)
        rows = frame.to_dict(orient="records")
    except Exception as exc:  # pragma: no cover - backend-specific error text
        raise SourceFormatError(f"Could not read parquet {path}: {exc}") from exc
    return [dict(row) for row in rows]


def _read_data_file(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix in {".parquet", ".pq"}:
        return _read_parquet(path)
    raise SourceFormatError(f"Unsupported source file type: {path}")


def _iter_file_rows(path: Path) -> Iterator[Tuple[Dict[str, Any], Path, Path]]:
    """Yield ``(row, row_base_dir, source_file)`` for a file or directory."""

    if path.is_file():
        for row in _read_data_file(path):
            yield row, path.parent, path
        return

    if not path.is_dir():
        raise SourceFormatError(f"Source path does not exist: {path}")

    files: List[Path] = []
    for pattern in ("*.jsonl", "*.json", "*.parquet", "*.pq"):
        files.extend(path.rglob(pattern))
    ignored_names = {"dataset_info.json", "state.json", "README.json"}
    for file_path in sorted(set(files)):
        if file_path.name in ignored_names:
            continue
        for row in _read_data_file(file_path):
            yield row, file_path.parent, file_path


def _iter_geometry_rows(path: Path) -> Iterator[Tuple[Dict[str, Any], Path, Path]]:
    """Read raw Geometry3K problem directories when present."""

    if path.is_dir():
        metadata_files = sorted(path.rglob("data.json"))
        if metadata_files:
            for metadata_file in metadata_files:
                rows = _read_json(metadata_file)
                for row in rows:
                    if not _image_values(row):
                        diagram = metadata_file.parent / "img_diagram.png"
                        if diagram.is_file():
                            row = dict(row)
                            row["images"] = [str(diagram)]
                    yield row, metadata_file.parent, metadata_file
            return
    elif path.is_file() and path.name == "data.json":
        rows = _read_json(path)
        for row in rows:
            if not _image_values(row):
                diagram = path.parent / "img_diagram.png"
                if diagram.is_file():
                    row = dict(row)
                    row["images"] = [str(diagram)]
            yield row, path.parent, path
        return
    yield from _iter_file_rows(path)


def _geoqa_split_ids(root: Path, split: str) -> Optional[set[str]]:
    """Read official GeoQA split IDs without copying embedded image arrays."""

    if split == "all":
        return None
    split_path = root / f"{split}.pk"
    if not split_path.is_file():
        raise SourceFormatError(f"GeoQA split file not found: {split_path}")
    try:
        with split_path.open("rb") as handle:
            rows = pickle.load(handle)
    except Exception as exc:
        raise SourceFormatError(f"Could not read GeoQA split {split_path}: {exc}") from exc
    return {
        str(row["id"])
        for row in rows
        if isinstance(row, dict) and row.get("id") is not None
    }


def _iter_geoqa_rows(path: Path, split: str = "train") -> Iterator[Tuple[Dict[str, Any], Path, Path]]:
    """Read the official GeoQA3 JSON/image layout and honor its split file."""

    root = path
    if root.name == "json" and root.is_dir():
        root = root.parent
    if (root / "GeoQA3").is_dir():
        root = root / "GeoQA3"
    json_dir = root / "json"
    image_dir = root / "image"
    if not json_dir.is_dir():
        raise SourceFormatError(f"GeoQA JSON directory not found: {json_dir}")
    allowed_ids = _geoqa_split_ids(root, split)
    files = sorted(
        json_dir.glob("*.json"),
        key=lambda item: int(item.stem) if item.stem.isdigit() else item.stem,
    )
    for metadata_file in files:
        if allowed_ids is not None and metadata_file.stem not in allowed_ids:
            continue
        for row in _read_json(metadata_file):
            row = dict(row)
            image_path = image_dir / f"{metadata_file.stem}.png"
            if image_path.is_file():
                row["images"] = [str(image_path)]
            yield row, root, metadata_file


def _extract_conversation_text(value: Any) -> Optional[str]:
    value = _unwrap_singleton(value)
    if isinstance(value, dict):
        return _as_text(value.get("content") or value.get("text"))
    if not isinstance(value, list):
        return _as_text(value)
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("from") or item.get("speaker") or "").lower()
        if role in {"user", "human", "question", "prompter"}:
            text = _as_text(item.get("content") or item.get("value") or item.get("text"))
            if text:
                return text
    for item in value:
        text = _as_text(item)
        if text:
            return text
    return None


def _extract_question(row: Dict[str, Any]) -> Optional[str]:
    # ReTool/verl rows store the question as prompt[0].content.
    for key in ("prompt", "messages", "conversations"):
        if key in row:
            text = _extract_conversation_text(row[key])
            if text:
                return text
    for key in _QUESTION_KEYS:
        text = _as_text(row.get(key))
        if text:
            return text
    return None


def _image_values(row: Dict[str, Any]) -> List[Any]:
    for key in _IMAGE_KEYS:
        if key not in row or row[key] is None:
            continue
        value = _unwrap_singleton(row[key])
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
    return []


def _resolve_image(value: Any, base_dir: Path) -> str:
    value = _unwrap_singleton(value)
    if isinstance(value, dict):
        value = value.get("path") or value.get("filename") or value.get("image_path")
    if not isinstance(value, str) or not value.strip():
        raise SourceFormatError(
            "Image values must be local paths, data URLs, or http(s) URLs; "
            f"got {type(value).__name__}"
        )
    value = value.strip()
    if value.startswith(("data:", "http://", "https://")):
        return value
    image_path = Path(value).expanduser()
    if image_path.is_absolute():
        candidates = [image_path]
    else:
        candidates = [
            base_dir / image_path,
            base_dir / "mulberry_images" / image_path,
            base_dir / "images" / image_path,
            base_dir.parent / image_path,
        ]
    image_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not image_path.exists() or not image_path.is_file():
        raise SourceFormatError(f"Image path does not exist: {image_path}")
    return str(image_path.resolve())


def _extract_images(row: Dict[str, Any], base_dir: Path) -> List[str]:
    return [_resolve_image(value, base_dir) for value in _image_values(row)]


def _extract_choices(row: Dict[str, Any]) -> List[str]:
    value = _unwrap_singleton(row.get("choices", row.get("options")))
    if isinstance(value, dict):
        value = [value[key] for key in sorted(value)]
    if not isinstance(value, (list, tuple)):
        return []
    return [text for item in value if (text := _as_text(item))]


def _question_with_choices(question: str, choices: Sequence[str]) -> str:
    if not choices:
        return question
    # Avoid appending a second copy when an exported dataset already embeds
    # the choices in its problem text.
    if all(choice in question for choice in choices):
        return question
    labels = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    return f"{question}\n\nChoices:\n{labels}"


def _clean_question(question: str) -> str:
    # The image marker belongs in the normalized conversation, not in the
    # source text.  The builder adds exactly one marker per resolved image.
    question = re.sub(r"<image>", "", question, flags=re.IGNORECASE)
    # Some source rows were serialized from a string containing ``\boxed``
    # without escaping the backslash, so ``\b`` became a backspace character.
    # Restore the intended LaTeX command before putting the question in SFT.
    question = question.replace("\x08oxed", r"\boxed")
    return re.sub(r"\n{3,}", "\n\n", question).strip()


def _raw_ground_truth(row: Dict[str, Any], source: str) -> Any:
    value: Any = None
    for key in ("ground_truth", "gt_answer", "reference_answer", "answer"):
        if key in row and row[key] not in (None, ""):
            value = row[key]
            break
    if value is None:
        value = _nested_value(row, "reward_model", "ground_truth")
    if value is None and source == "mulberry":
        value = row.get("gt")
    if value is None and source in {"mulberry", "retool"}:
        messages = _unwrap_singleton(row.get("messages"))
        if isinstance(messages, (list, tuple)):
            for message in reversed(messages):
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or message.get("from") or "").lower()
                if role in {"assistant", "gpt", "bot"}:
                    value = message.get("content")
                    if value:
                        break
    return value


def _reference_marker(text: str, source: str) -> str:
    """Extract a final marker from sources that contain old assistant turns."""

    if source not in {"mulberry", "retool"}:
        return text
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    final = re.findall(r"FINAL_ANSWER:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
    if final:
        return final[-1].strip()
    answer = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer:
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", answer[-1])
        return boxed[-1].strip() if boxed else answer[-1].strip()
    final = re.findall(
        r"(?:the\s+final\s+answer\s+is|final\s+answer\s*:)\s*:?\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if final:
        return final[-1].strip().rstrip(".。")
    return text


def _ground_truth_candidates(
    row: Dict[str, Any], choices: Sequence[str], source: str
) -> List[str]:
    if source == "geoqa" and isinstance(row.get("label"), (int, float)):
        label = int(row["label"])
        if 0 <= label < len(choices):
            return [str(choices[label]), chr(ord("A") + label)]
    value = _unwrap_singleton(_raw_ground_truth(row, source))
    text = _as_text(value)
    if not text:
        return []
    text = _reference_marker(text, source)

    candidates = [text]
    # Preserve both representations for multiple-choice data. Matching a
    # model's exact ``C`` against the exact text of choice C is deterministic,
    # unlike fuzzy matching, and keeps the source's original answer intact.
    if choices and re.fullmatch(r"[A-Za-z]", text):
        index = ord(text.upper()) - ord("A")
        if 0 <= index < len(choices):
            candidates.append(choices[index])
    return candidates


def _extract_ground_truth(row: Dict[str, Any], choices: Sequence[str], source: str) -> Optional[str]:
    candidates = _ground_truth_candidates(row, choices, source)
    return candidates[-1] if candidates else None


def _task_id(row: Dict[str, Any], source_file: Path, ordinal: int, source: str) -> str:
    for key in ("task_id", "problem_id", "pid", "id", "index"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    extra_index = _nested_value(row, "extra_info", "index")
    if extra_index not in (None, ""):
        return str(extra_index)
    return f"{source}:{source_file}:{ordinal}"


def _normalize_row(
    row: Dict[str, Any],
    base_dir: Path,
    source_file: Path,
    ordinal: int,
    source: str,
    stage: int,
) -> Dict[str, Any]:
    question = _extract_question(row)
    if not question:
        raise SourceFormatError(f"No question found in {source_file} row {ordinal}")
    choices = _extract_choices(row)
    if source == "geometry3k":
        question = _question_with_choices(question, choices)
    images = _extract_images(row, base_dir)
    question = _clean_question(question)
    if not question:
        raise SourceFormatError(f"Question is empty after normalization in {source_file} row {ordinal}")
    ground_truth_candidates = _ground_truth_candidates(row, choices, source)
    return {
        "task_id": _task_id(row, source_file, ordinal, source),
        "source_dataset": source,
        "stage": stage,
        "question": question,
        "images": images,
        "ground_truth": _extract_ground_truth(row, choices, source),
        "ground_truth_aliases": ground_truth_candidates,
    }


def iter_source_samples(
    source: str,
    source_path: str | Path,
    stage: Optional[int] = None,
    source_split: str = "train",
) -> Iterator[Dict[str, Any]]:
    """Yield normalized source samples.

    The stage is a property of the source adapter: Geometry3K/Mulberry are
    Stage 1 sources and ReTool is a Stage 2 source.  Passing a contradictory
    stage is rejected rather than silently randomizing or reassigning data.
    """

    canonical = canonical_source_name(source)
    expected_stage = SOURCE_STAGES[canonical]
    allowed_stages = SOURCE_STAGE_OPTIONS[canonical]
    active_stage = expected_stage if stage is None else int(stage)
    if active_stage not in allowed_stages:
        allowed = ", ".join(str(item) for item in allowed_stages)
        raise SourceFormatError(f"Source {canonical!r} supports Stage {allowed}, not Stage {active_stage}")

    path = Path(source_path).expanduser()
    if canonical == "geometry3k":
        row_iter = _iter_geometry_rows(path)
    elif canonical == "geoqa":
        row_iter = _iter_geoqa_rows(path, split=source_split)
    else:
        row_iter = _iter_file_rows(path)
    for ordinal, (row, base_dir, source_file) in enumerate(row_iter):
        yield _normalize_row(row, base_dir, source_file, ordinal, canonical, active_stage)


def load_source(
    source: str,
    source_path: str | Path,
    stage: Optional[int] = None,
    source_split: str = "train",
) -> List[Dict[str, Any]]:
    """Materialize :func:`iter_source_samples` for one-shot inspection/tests."""

    return list(iter_source_samples(source, source_path, stage=stage, source_split=source_split))


def format_user_question(question: str, images: Sequence[str]) -> str:
    """Create the exact user text expected by the released solver runtime."""

    clean = _clean_question(str(question))
    if not images:
        return clean
    markers = "\n".join("<image>" for _ in images)
    return f"{markers}\n{clean}" if clean else markers


__all__ = [
    "SOURCE_STAGES",
    "SourceFormatError",
    "canonical_source_name",
    "format_user_question",
    "iter_source_samples",
    "load_source",
]
