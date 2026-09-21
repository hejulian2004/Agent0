"""Exact, order-preserving deduplication for exported SFT rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List


def record_hash(record: Dict[str, Any]) -> str:
    """Hash only the trainable ``messages`` and ``images`` payload."""

    canonical = json.dumps(
        {"messages": record.get("messages", []), "images": record.get("images", [])},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def exact_deduplicate(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the first row for each exact messages/images hash."""

    result: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        digest = record_hash(record)
        if digest in seen:
            continue
        seen.add(digest)
        result.append(record)
    return result


__all__ = ["exact_deduplicate", "record_hash"]
