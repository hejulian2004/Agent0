"""Fail-closed source partition and leakage checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_bytes, normalize_text, sha256_json


FORBIDDEN_PARTITIONS = frozenset({"test", "validation", "eval", "rl", "reserve"})
ALLOWED_USAGE = frozenset({"sft_stage1", "sft_stage2"})


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
        task_id = record.get("task_id") or record.get("id") or record.get("sample_id")
        source_dataset = record.get("source_dataset") or record.get("data_source")
        source_revision = record.get("source_revision") or record.get("revision") or "unknown"
        original_id = record.get("original_id") or record.get("source_record_id") or task_id
        if not task_id or not source_dataset or not original_id:
            raise ValueError("source identity requires task_id, source_dataset and original_id")

        image_hashes = tuple(
            normalize_text(str(item)).lower()
            for item in (
                record.get("image_content_hashes")
                or record.get("image_sha256")
                or record.get("image_hashes")
                or []
            )
        )
        source_content_hash = record.get("source_content_hash")
        if not source_content_hash:
            source_content_hash = sha256_json(
                {
                    "question": record.get("question") or record.get("prompt") or "",
                    "image_content_hashes": list(image_hashes),
                }
            )
        return cls(
            task_id=normalize_text(str(task_id)),
            source_dataset=normalize_text(str(source_dataset)),
            source_revision=normalize_text(str(source_revision)),
            original_id=normalize_text(str(original_id)),
            source_content_hash=normalize_text(str(source_content_hash)).lower(),
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
        if split is None or usage_partition is None:
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
