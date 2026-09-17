"""Deterministic serialization helpers shared by snapshots and deduplication."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    """Normalize Unicode without changing whitespace or code semantics."""

    return unicodedata.normalize("NFC", value)


def normalize_json_value(value: Any) -> Any:
    """Recursively normalize strings while preserving list order and content."""

    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, dict):
        return {
            normalize_text(str(key)): normalize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, set):
        raise TypeError("sets are not valid canonical JSON values")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository-wide canonical UTF-8 JSON representation."""

    normalized = normalize_json_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
