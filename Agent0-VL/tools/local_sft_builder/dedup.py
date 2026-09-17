"""Versioned deterministic exact deduplication for final SFT rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .canonical import canonical_json_bytes, sha256_bytes
from .schema import DEDUP_KEY_VERSION, ExportRow


class DedupError(ValueError):
    pass


def exact_dedup_key(row: ExportRow) -> str:
    if len(row.images) != len(row.image_content_hashes):
        raise DedupError("each image needs an ordered content SHA256")
    payload = {
        "dedup_key_version": DEDUP_KEY_VERSION,
        "messages": list(row.messages),
        "image_content_sha256": list(row.image_content_hashes),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _winner_key(row: ExportRow) -> tuple[int, str, str, str]:
    stage_order = {"sft_stage1": 0, "sft_stage2": 1}
    return (
        stage_order.get(row.stage, 99),
        row.source_record_id,
        row.trajectory_id,
        exact_dedup_key(row),
    )


def exact_deduplicate(rows: Iterable[ExportRow]) -> list[ExportRow]:
    groups: dict[str, list[ExportRow]] = defaultdict(list)
    for row in rows:
        groups[exact_dedup_key(row)].append(row)
    winners = [min(group, key=_winner_key) for group in groups.values()]
    winners.sort(key=_winner_key)
    return winners
