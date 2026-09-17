"""Fail-closed source partition and leakage checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical import normalize_text, sha256_json


FORBIDDEN_PARTITIONS = frozenset({"test", "validation", "eval", "rl", "reserve"})
ALLOWED_USAGE = frozenset({"sft_stage1", "sft_stage2"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DECLARATION_FIELDS = ("images", "image_refs")


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase SHA256 hex digest"
        )
    return value


def _declared_image_count(record: Mapping[str, Any]) -> int | None:
    counts: list[int] = []

    if "image_count" in record:
        image_count = record["image_count"]
        if isinstance(image_count, bool) or not isinstance(image_count, int):
            raise ValueError("image_count must be a non-negative integer")
        if image_count < 0:
            raise ValueError("image_count must be a non-negative integer")
        counts.append(image_count)

    for field_name in _IMAGE_DECLARATION_FIELDS:
        if field_name not in record:
            continue
        values = record[field_name]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{field_name} must be a list or tuple")
        counts.append(len(values))

    if not counts:
        return None
    if len(set(counts)) != 1:
        raise ValueError("image declarations have inconsistent cardinality")
    return counts[0]


def _image_hashes(record: Mapping[str, Any]) -> tuple[str, ...]:
    raw_hashes: Any = None
    found = False
    for field_name in ("image_content_hashes", "image_sha256", "image_hashes"):
        if field_name in record:
            raw_hashes = record[field_name]
            found = True
            break

    if not found:
        return ()
    if raw_hashes is None:
        raise ValueError("image content hashes must not be null")
    if not isinstance(raw_hashes, (list, tuple)):
        raise ValueError("image content hashes must be a list or tuple")
    return tuple(
        _require_sha256(item, field_name="image_content_hash")
        for item in raw_hashes
    )


@dataclass(frozen=True)
class SourceIdentity:
    task_id: str
    source_dataset: str
    source_revision: str
    original_id: str
    source_content_hash: str
    image_content_hashes: tuple[str, ...] = ()

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SourceIdentity":
        identity_fields = (
            "task_id",
            "source_dataset",
            "source_revision",
            "original_id",
        )
        for field_name in identity_fields:
            value = record.get(field_name)
            if not isinstance(value, str) or not normalize_text(value).strip():
                raise ValueError(f"source identity requires {field_name}")
        task_id = record["task_id"]
        source_dataset = record["source_dataset"]
        source_revision = record["source_revision"]
        original_id = record["original_id"]
        source_revision_text = normalize_text(source_revision).strip()
        if source_revision_text.casefold() == "unknown":
            raise ValueError("source identity requires source_revision")

        image_hashes = _image_hashes(record)
        image_count = _declared_image_count(record)
        if image_count is not None and image_count != len(image_hashes):
            raise ValueError(
                "image path/reference count must equal image hash count"
            )

        if "source_content_hash" in record:
            source_content_hash = record["source_content_hash"]
        else:
            source_content_hash = sha256_json(
                {
                    "question": record.get("question") or record.get("prompt") or "",
                    "image_content_hashes": list(image_hashes),
                }
            )
        source_content_hash = _require_sha256(
            source_content_hash,
            field_name="source_content_hash",
        )
        return cls(
            task_id=normalize_text(str(task_id)),
            source_dataset=normalize_text(str(source_dataset)),
            source_revision=source_revision_text,
            original_id=normalize_text(str(original_id)),
            source_content_hash=normalize_text(str(source_content_hash)),
            image_content_hashes=image_hashes,
        )

    def keys(self) -> frozenset[tuple[str, str]]:
        keys: set[tuple[str, str]] = {
            ("task_id", self.task_id),
            (
                "dataset_revision_original_id",
                "|".join((self.source_dataset, self.source_revision, self.original_id)),
            ),
            ("source_content_hash", self.source_content_hash),
        }
        keys.update(("image_content_hash", item) for item in self.image_content_hashes)
        return frozenset(keys)


@dataclass(frozen=True)
class SourceGuardDecision:
    status: str
    reasons: tuple[str, ...] = ()
    identity: SourceIdentity | None = None

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


class SourceLeakageGuard:
    """Guard train candidates against forbidden partitions before generation."""

    def __init__(self, forbidden_records: Iterable[Mapping[str, Any]] = ()):
        self._forbidden_keys: set[tuple[str, str]] = set()
        for record in forbidden_records:
            identity = SourceIdentity.from_record(record)
            self._forbidden_keys.update(identity.keys())

    def check(
        self,
        record: Mapping[str, Any],
        *,
        expected_stage: str | None = None,
    ) -> SourceGuardDecision:
        split = record.get("split")
        usage_partition = record.get("usage_partition")
        if (
            not isinstance(split, str)
            or not split.strip()
            or not isinstance(usage_partition, str)
            or not usage_partition.strip()
        ):
            return SourceGuardDecision(
                "review_required",
                ("missing_or_unknown_split_metadata",),
            )
        split = normalize_text(str(split)).lower()
        usage_partition = normalize_text(str(usage_partition)).lower()
        if split in FORBIDDEN_PARTITIONS or usage_partition in FORBIDDEN_PARTITIONS:
            return SourceGuardDecision(
                "rejected",
                ("forbidden_partition",),
            )
        if split != "train":
            return SourceGuardDecision("rejected", ("split_is_not_train",))
        if usage_partition not in ALLOWED_USAGE:
            return SourceGuardDecision("rejected", ("usage_partition_not_sft",))
        if expected_stage is not None and usage_partition != expected_stage:
            return SourceGuardDecision("rejected", ("stage_partition_mismatch",))

        try:
            identity = SourceIdentity.from_record(record)
        except (TypeError, ValueError) as exc:
            return SourceGuardDecision("review_required", (f"invalid_source_identity:{exc}",))
        overlap = identity.keys().intersection(self._forbidden_keys)
        if overlap:
            return SourceGuardDecision(
                "rejected",
                ("forbidden_source_identity_overlap",),
                identity,
            )
        return SourceGuardDecision("accepted", identity=identity)

    def assert_train_candidate(
        self,
        record: Mapping[str, Any],
        *,
        expected_stage: str,
    ) -> SourceIdentity:
        decision = self.check(record, expected_stage=expected_stage)
        if not decision.accepted or decision.identity is None:
            raise ValueError(
                "source guard rejected candidate: " + ",".join(decision.reasons)
            )
        return decision.identity
