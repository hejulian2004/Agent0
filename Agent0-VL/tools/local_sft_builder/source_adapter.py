"""Strict normalized-source loading and full preflight for Phase 2A.

The adapter is deliberately independent from the formal data builder.  It
accepts only a normalized JSONL contract, validates every input row before
sampling, and returns ``SourceTask`` objects only for rows accepted by the
source guard.  No Teacher, snapshot, trajectory, or export code is invoked
from this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import normalize_text, sha256_file
from .fake_builder import SourceTask
from .source_guard import SourceGuardDecision, SourceIdentity, SourceLeakageGuard


SOURCE_SCHEMA_VERSION = "agent0vl.local_sft_builder.source.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:")
_STAGE_PARTITIONS = {
    "stage1": "sft_stage1",
    "stage2": "sft_stage2",
    "sft_stage1": "sft_stage1",
    "sft_stage2": "sft_stage2",
}
_REQUIRED_SOURCE_FIELDS = (
    "task_id",
    "source_record_id",
    "original_id",
    "source_dataset",
    "source_revision",
    "split",
    "usage_partition",
    "question",
    "ground_truth",
    "images",
    "image_refs",
)


class SourcePreflightError(RuntimeError):
    """Base class for errors that invalidate the complete preflight."""


class SourceInputError(SourcePreflightError):
    """Raised for unreadable or structurally malformed JSONL input."""


class ForbiddenIndexError(SourcePreflightError):
    """Raised when a forbidden identity index cannot be trusted."""


class DuplicateSourceIdentityError(SourcePreflightError):
    """Raised when a stable identity occurs more than once."""


class InvalidImageRootError(SourcePreflightError):
    """Raised when image-root configuration is invalid for the input."""


class _RowReviewRequired(ValueError):
    """A data-quality issue that affects one row but not the whole scan."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _stage_partition(stage: str) -> str:
    if not isinstance(stage, str) or stage not in _STAGE_PARTITIONS:
        raise ValueError("stage must be stage1, stage2, sft_stage1 or sft_stage2")
    return _STAGE_PARTITIONS[stage]


def _required_text(row: Mapping[str, Any], field_name: str) -> str:
    if field_name not in row:
        raise _RowReviewRequired(f"missing_field:{field_name}")
    value = row[field_name]
    if not isinstance(value, str):
        raise _RowReviewRequired(f"{field_name}_must_be_string")
    value = normalize_text(value).strip()
    if not value:
        raise _RowReviewRequired(f"{field_name}_must_be_non_empty")
    return value


def _required_revision(row: Mapping[str, Any]) -> str:
    revision = _required_text(row, "source_revision")
    if revision.casefold() == "unknown":
        raise _RowReviewRequired("source_revision_must_be_known")
    return revision


def _required_content(row: Mapping[str, Any], field_name: str) -> str:
    if field_name not in row:
        raise _RowReviewRequired(f"missing_field:{field_name}")
    value = row[field_name]
    if not isinstance(value, str):
        raise _RowReviewRequired(f"{field_name}_must_be_string")
    return normalize_text(value)


def _strict_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _RowReviewRequired(f"invalid_{field_name}")
    return value


def _is_absolute_like(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        normalized.startswith("/")
        or normalized.startswith("//")
        or bool(_WINDOWS_ABSOLUTE_RE.match(normalized))
    )


def _logical_asset_id(value: Any, index: int) -> str:
    if not isinstance(value, str):
        raise _RowReviewRequired(f"invalid_asset_id:{index}")
    asset_id = normalize_text(value).strip().replace("\\", "/")
    if not asset_id or _is_absolute_like(asset_id):
        raise _RowReviewRequired(f"invalid_asset_id:{index}")
    if ".." in asset_id.split("/"):
        raise _RowReviewRequired(f"invalid_asset_id:{index}")
    return asset_id


def _image_hashes_from_refs(
    image_refs: Sequence[Any],
    *,
    field_prefix: str = "image_ref",
) -> tuple[dict[str, str], ...]:
    normalized: list[dict[str, str]] = []
    for index, raw_ref in enumerate(image_refs):
        if not isinstance(raw_ref, Mapping):
            raise _RowReviewRequired(f"{field_prefix}_must_be_mapping:{index}")
        raw_asset_id = (
            raw_ref["asset_id"]
            if "asset_id" in raw_ref
            else raw_ref.get("logical_asset_id")
        )
        asset_id = _logical_asset_id(raw_asset_id, index)
        content_hash = _strict_sha256(
            raw_ref.get("content_sha256"),
            f"image_content_sha256:{index}",
        )
        normalized.append(
            {
                "asset_id": asset_id,
                "content_sha256": content_hash,
            }
        )
    return tuple(normalized)


def _validate_image_hash_list(
    value: Any,
    *,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _RowReviewRequired(f"{field_name}_must_be_list")
    return tuple(
        _strict_sha256(item, f"{field_name}:{index}")
        for index, item in enumerate(value)
    )


def _source_preview(row: Mapping[str, Any], field_name: str) -> str | None:
    value = row.get(field_name)
    if isinstance(value, str):
        return normalize_text(value).strip() or None
    return None


def _optional_required_text(
    row: Mapping[str, Any],
    field_name: str,
    *,
    revision: bool = False,
) -> str | None:
    try:
        return _required_revision(row) if revision else _required_text(row, field_name)
    except _RowReviewRequired:
        return None


def _stable_identity_keys(
    row: Mapping[str, Any],
) -> tuple[str | None, tuple[str, str, str] | None, tuple[str, str] | None]:
    task_id = _optional_required_text(row, "task_id")
    source_dataset = _optional_required_text(row, "source_dataset")
    source_record_id = _optional_required_text(row, "source_record_id")
    original_id = _optional_required_text(row, "original_id")
    source_revision = _optional_required_text(row, "source_revision", revision=True)
    dataset_original = (
        (source_dataset, source_revision, original_id)
        if source_dataset is not None
        and source_revision is not None
        and original_id is not None
        else None
    )
    dataset_record = (
        (source_dataset, source_record_id)
        if source_dataset is not None and source_record_id is not None
        else None
    )
    return task_id, dataset_original, dataset_record


def _read_jsonl(paths: Sequence[str | Path], *, error_type: type[SourcePreflightError]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    seen_paths: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            resolved_path = path.resolve()
        except OSError as exc:
            raise error_type(f"cannot resolve input file: {path}") from exc
        if resolved_path in seen_paths:
            raise error_type(f"duplicate input file: {resolved_path}")
        seen_paths.add(resolved_path)
        if not resolved_path.is_file():
            raise error_type(f"input file is not readable: {resolved_path}")
        try:
            handle = resolved_path.open("r", encoding="utf-8")
        except OSError as exc:
            raise error_type(f"cannot read input file: {resolved_path}") from exc
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise error_type(
                        f"malformed JSONL at {resolved_path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise error_type(
                        f"JSONL row is not an object at {resolved_path}:{line_number}"
                    )
                yield resolved_path, line_number, value


@dataclass(frozen=True)
class SourcePreflightEntry:
    """Serializable decision for exactly one source row."""

    source_path: str
    line_number: int
    status: str
    reasons: tuple[str, ...] = ()
    task_id: str | None = None
    source_record_id: str | None = None
    original_id: str | None = None
    identity: SourceIdentity | None = None
    image_content_hashes: tuple[str, ...] = ()
    source_content_hash_origin: str | None = None
    selected: bool = False
    task: SourceTask | None = None

    def to_dict(self) -> dict[str, Any]:
        identity = None
        if self.identity is not None:
            identity = {
                "task_id": self.identity.task_id,
                "source_dataset": self.identity.source_dataset,
                "source_revision": self.identity.source_revision,
                "original_id": self.identity.original_id,
                "source_content_hash": self.identity.source_content_hash,
            }
        return {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_path": self.source_path,
            "line_number": self.line_number,
            "task_id": self.task_id,
            "source_record_id": self.source_record_id,
            "original_id": self.original_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "identity": identity,
            "image_content_hashes": list(self.image_content_hashes),
            "source_content_hash_origin": self.source_content_hash_origin,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class SourcePreflightResult:
    """All row decisions plus the accepted and selected task pools."""

    entries: tuple[SourcePreflightEntry, ...]
    accepted_tasks: tuple[SourceTask, ...]
    selected_tasks: tuple[SourceTask, ...]
    seed: int
    max_tasks: int | None

    @property
    def total_source_rows(self) -> int:
        return len(self.entries)

    @property
    def accepted_before_sampling(self) -> int:
        return len(self.accepted_tasks)

    @property
    def selected_after_sampling(self) -> int:
        return len(self.selected_tasks)

    @property
    def counts(self) -> dict[str, int]:
        counts = {"accepted": 0, "rejected": 0, "review_required": 0}
        for entry in self.entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return counts


@dataclass(frozen=True)
class ForbiddenSourceIndex:
    """Identity-only index for sources that must never enter SFT."""

    identities: tuple[SourceIdentity, ...] = ()
    keys: frozenset[tuple[str, str]] = frozenset()

    @classmethod
    def empty(cls) -> "ForbiddenSourceIndex":
        return cls()

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path]) -> "ForbiddenSourceIndex":
        identities: list[SourceIdentity] = []
        keys: set[tuple[str, str]] = set()
        seen_task_ids: set[str] = set()
        seen_dataset_originals: set[tuple[str, str, str]] = set()
        seen_dataset_records: set[tuple[str, str]] = set()
        for path, line_number, row in _read_jsonl(
            paths,
            error_type=ForbiddenIndexError,
        ):
            try:
                canonical = _canonical_forbidden_record(row)
                identity = SourceIdentity.from_record(canonical)
            except (TypeError, ValueError, _RowReviewRequired) as exc:
                raise ForbiddenIndexError(
                    f"invalid forbidden identity at {path}:{line_number}: {exc}"
                ) from exc
            dataset_original = (
                identity.source_dataset,
                identity.source_revision,
                identity.original_id,
            )
            source_record_id = canonical.get("source_record_id")
            dataset_record = (
                (identity.source_dataset, source_record_id)
                if source_record_id is not None
                else None
            )
            if (
                identity.task_id in seen_task_ids
                or dataset_original in seen_dataset_originals
                or (dataset_record is not None and dataset_record in seen_dataset_records)
            ):
                raise ForbiddenIndexError(
                    f"duplicate forbidden identity at {path}:{line_number}"
                )
            seen_task_ids.add(identity.task_id)
            seen_dataset_originals.add(dataset_original)
            if dataset_record is not None:
                seen_dataset_records.add(dataset_record)
            identities.append(identity)
            keys.update(identity.keys())
        return cls(tuple(identities), frozenset(keys))

    @property
    def record_count(self) -> int:
        return len(self.identities)

    def overlaps(self, identity: SourceIdentity) -> frozenset[tuple[str, str]]:
        return frozenset(identity.keys().intersection(self.keys))


def _canonical_forbidden_record(row: Mapping[str, Any]) -> dict[str, Any]:
    for field_name in ("task_id", "source_dataset", "source_revision", "original_id"):
        _required_text(row, field_name)

    canonical = dict(row)
    canonical["task_id"] = _required_text(row, "task_id")
    canonical["source_dataset"] = _required_text(row, "source_dataset")
    canonical["source_revision"] = _required_revision(row)
    canonical["original_id"] = _required_text(row, "original_id")
    if "source_record_id" in row:
        canonical["source_record_id"] = _required_text(row, "source_record_id")

    if "question" in row:
        if not isinstance(row["question"], str):
            raise _RowReviewRequired("question_must_be_string")
        canonical["question"] = normalize_text(row["question"])
    elif "source_content_hash" not in row:
        raise _RowReviewRequired("missing_question_or_source_content_hash")

    if "source_content_hash" in row:
        canonical["source_content_hash"] = _strict_sha256(
            row["source_content_hash"],
            "source_content_hash",
        )

    image_hashes: tuple[str, ...] = ()
    if "image_refs" in row:
        raw_refs = row["image_refs"]
        if not isinstance(raw_refs, list):
            raise _RowReviewRequired("image_refs_must_be_list")
        refs = _image_hashes_from_refs(raw_refs)
        image_hashes = tuple(ref["content_sha256"] for ref in refs)
        canonical["image_refs"] = list(refs)

    if "image_content_hashes" in row:
        declared_hashes = _validate_image_hash_list(
            row["image_content_hashes"],
            field_name="image_content_hashes",
        )
        if image_hashes and declared_hashes != image_hashes:
            raise _RowReviewRequired("image_hash_declarations_disagree")
        image_hashes = declared_hashes

    if "images" in row:
        if not isinstance(row["images"], list):
            raise _RowReviewRequired("images_must_be_list")
        if len(row["images"]) != len(image_hashes):
            raise _RowReviewRequired("image_hash_count_mismatch")
        canonical["image_count"] = len(row["images"])
    elif "image_refs" in row and len(image_hashes):
        canonical["image_count"] = len(image_hashes)

    canonical["image_content_hashes"] = list(image_hashes)
    return canonical


class RealSourceAdapter:
    """Read normalized source JSONL and perform a complete source preflight."""

    def __init__(
        self,
        source_paths: Sequence[str | Path] | str | Path,
        *,
        stage: str,
        image_root: str | Path | None = None,
        image_root_config: str | None = None,
        source_guard: SourceLeakageGuard | None = None,
        forbidden_index: ForbiddenSourceIndex | None = None,
    ):
        if isinstance(source_paths, (str, Path)):
            source_paths = (source_paths,)
        self.source_paths = tuple(Path(path).expanduser() for path in source_paths)
        if not self.source_paths:
            raise ValueError("at least one source path is required")
        self.stage = _stage_partition(stage)
        self.source_guard = source_guard or SourceLeakageGuard()
        self.forbidden_index = forbidden_index or ForbiddenSourceIndex.empty()

        self.image_root: Path | None
        if image_root is None:
            self.image_root = None
        else:
            root = Path(image_root).expanduser()
            try:
                resolved_root = root.resolve()
            except OSError as exc:
                raise InvalidImageRootError(f"cannot resolve image_root: {root}") from exc
            if not resolved_root.is_dir():
                raise InvalidImageRootError(
                    f"image_root must be an existing directory: {resolved_root}"
                )
            self.image_root = resolved_root

        if image_root_config is None:
            self.image_root_config = None
        else:
            if not isinstance(image_root_config, str):
                raise InvalidImageRootError("image_root_config must be a string")
            config = normalize_text(image_root_config).strip().replace("\\", "/")
            if not config or _is_absolute_like(config) or ".." in config.split("/"):
                raise InvalidImageRootError("image_root_config must be a relative logical identifier")
            self.image_root_config = config

        if self.image_root is not None and self.image_root_config is None:
            raise InvalidImageRootError(
                "image_root_config is required when image_root is configured"
            )
        if self.image_root is None and self.image_root_config is not None:
            raise InvalidImageRootError(
                "image_root requires a physical image_root directory"
            )

    def _resolve_image(self, value: Any, index: int) -> tuple[str, str]:
        if not isinstance(value, str) or not value.strip():
            raise _RowReviewRequired(f"invalid_image_path:{index}")
        relative_value = normalize_text(value).strip().replace("\\", "/")
        if _is_absolute_like(relative_value):
            raise _RowReviewRequired(f"absolute_image_path:{index}")
        if ".." in relative_value.split("/"):
            raise _RowReviewRequired(f"image_path_traversal:{index}")
        if self.image_root is None:
            raise _RowReviewRequired(f"image_root_missing:{index}")
        try:
            candidate = (self.image_root / Path(relative_value)).resolve()
        except (OSError, RuntimeError) as exc:
            raise _RowReviewRequired(f"image_path_unresolvable:{index}") from exc
        try:
            candidate.relative_to(self.image_root)
        except ValueError as exc:
            raise _RowReviewRequired(f"image_path_outside_root:{index}") from exc
        try:
            is_file = candidate.is_file()
        except OSError as exc:
            raise _RowReviewRequired(f"image_unreadable:{index}") from exc
        if not is_file:
            raise _RowReviewRequired(f"image_missing:{index}")
        try:
            actual_hash = sha256_file(candidate)
        except OSError as exc:
            raise _RowReviewRequired(f"image_unreadable:{index}") from exc
        return str(candidate), actual_hash

    def _normalize_row(
        self,
        row: Mapping[str, Any],
    ) -> tuple[dict[str, Any], SourceTask, SourceIdentity, str, tuple[str, ...]]:
        missing = next(
            (field_name for field_name in _REQUIRED_SOURCE_FIELDS if field_name not in row),
            None,
        )
        if missing is not None:
            raise _RowReviewRequired(f"missing_field:{missing}")

        task_id = _required_text(row, "task_id")
        source_record_id = _required_text(row, "source_record_id")
        original_id = _required_text(row, "original_id")
        source_dataset = _required_text(row, "source_dataset")
        source_revision = _required_revision(row)
        split = _required_text(row, "split")
        usage_partition = _required_text(row, "usage_partition")
        question = _required_content(row, "question")
        if not question.strip():
            raise _RowReviewRequired("question_must_be_non_empty")
        ground_truth = _required_content(row, "ground_truth")

        images = row["images"]
        image_refs = row["image_refs"]
        if not isinstance(images, list):
            raise _RowReviewRequired("images_must_be_list")
        if not isinstance(image_refs, list):
            raise _RowReviewRequired("image_refs_must_be_list")
        if len(images) != len(image_refs):
            raise _RowReviewRequired("image_and_ref_count_mismatch")

        refs = _image_hashes_from_refs(image_refs)
        image_hashes = tuple(ref["content_sha256"] for ref in refs)
        if "image_count" in row:
            image_count = row["image_count"]
            if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count < 0:
                raise _RowReviewRequired("invalid_image_count")
            if image_count != len(images):
                raise _RowReviewRequired("image_count_mismatch")

        physical_images: list[str] = []
        actual_hashes: list[str] = []
        relative_images: list[str] = []
        for index, image in enumerate(images):
            physical_path, actual_hash = self._resolve_image(image, index)
            declared_hash = image_hashes[index]
            if actual_hash != declared_hash:
                raise _RowReviewRequired(f"image_hash_mismatch:{index}")
            physical_images.append(physical_path)
            actual_hashes.append(actual_hash)
            # ``_resolve_image`` already validated and normalized this value;
            # keep the portable form so the exported row stays image-root
            # relative instead of leaking a machine path.
            relative_images.append(
                normalize_text(str(image)).strip().replace("\\", "/")
            )

        if question.count("<image>") != len(images):
            raise _RowReviewRequired("image_placeholder_count_mismatch")

        source_content_hash_origin = "derived"
        canonical = dict(row)
        if "source_content_hash" in row:
            canonical["source_content_hash"] = _strict_sha256(
                row["source_content_hash"],
                "source_content_hash",
            )
            source_content_hash_origin = "declared"
        canonical.update(
            {
                "task_id": task_id,
                "source_record_id": source_record_id,
                "original_id": original_id,
                "source_dataset": source_dataset,
                "source_revision": source_revision,
                "split": split,
                "usage_partition": usage_partition,
                "question": question,
                "ground_truth": ground_truth,
                "images": physical_images,
                "image_refs": list(refs),
                "image_content_hashes": list(image_hashes),
                "image_count": len(physical_images),
            }
        )

        try:
            identity = SourceIdentity.from_record(canonical)
        except (TypeError, ValueError) as exc:
            raise _RowReviewRequired(f"invalid_source_identity:{exc}") from exc
        task = SourceTask(
            task_id=task_id,
            source_record_id=source_record_id,
            original_id=original_id,
            source_dataset=source_dataset,
            stage=self.stage,
            question=question,
            images=tuple(physical_images),
            image_refs=refs,
            image_relatives=tuple(relative_images),
            ground_truth=ground_truth,
            split=split,
            usage_partition=usage_partition,
            source_revision=source_revision,
        )
        return canonical, task, identity, source_content_hash_origin, tuple(actual_hashes)

    @staticmethod
    def _rank(seed: int, task_id: str) -> str:
        material = f"{seed}\0{task_id}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def preflight(
        self,
        *,
        seed: int = 0,
        max_tasks: int | None = None,
    ) -> SourcePreflightResult:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if max_tasks is not None and (
            isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or max_tasks <= 0
        ):
            raise ValueError("max_tasks must be a positive integer or None")

        entries: list[SourcePreflightEntry] = []
        accepted: list[tuple[SourcePreflightEntry, SourceTask]] = []
        seen_task_ids: set[str] = set()
        seen_dataset_originals: set[tuple[str, str, str]] = set()
        seen_dataset_records: set[tuple[str, str]] = set()

        for path, line_number, row in _read_jsonl(
            self.source_paths,
            error_type=SourceInputError,
        ):
            task_id = _source_preview(row, "task_id")
            source_record_id = _source_preview(row, "source_record_id")
            original_id = _source_preview(row, "original_id")
            stable_task_id, dataset_original, dataset_record = _stable_identity_keys(row)
            if (
                (stable_task_id is not None and stable_task_id in seen_task_ids)
                or (
                    dataset_original is not None
                    and dataset_original in seen_dataset_originals
                )
                or (
                    dataset_record is not None
                    and dataset_record in seen_dataset_records
                )
            ):
                raise DuplicateSourceIdentityError(
                    f"duplicate stable identity at {path}:{line_number}"
                )
            if stable_task_id is not None:
                seen_task_ids.add(stable_task_id)
            if dataset_original is not None:
                seen_dataset_originals.add(dataset_original)
            if dataset_record is not None:
                seen_dataset_records.add(dataset_record)
            try:
                canonical, task, identity, hash_origin, actual_hashes = self._normalize_row(row)

                overlap = self.forbidden_index.overlaps(identity)
                if overlap:
                    decision = SourceGuardDecision(
                        "rejected",
                        ("forbidden_source_identity_overlap",),
                        identity,
                    )
                else:
                    decision = self.source_guard.check(
                        canonical,
                        expected_stage=self.stage,
                    )
                entry = SourcePreflightEntry(
                    source_path=str(path),
                    line_number=line_number,
                    status=decision.status,
                    reasons=decision.reasons,
                    task_id=task.task_id,
                    source_record_id=task.source_record_id,
                    original_id=task.original_id,
                    identity=identity,
                    image_content_hashes=tuple(actual_hashes),
                    source_content_hash_origin=hash_origin,
                    task=task if decision.accepted else None,
                )
                entries.append(entry)
                if decision.accepted:
                    accepted.append((entry, task))
            except DuplicateSourceIdentityError:
                raise
            except InvalidImageRootError:
                raise
            except _RowReviewRequired as exc:
                entries.append(
                    SourcePreflightEntry(
                        source_path=str(path),
                        line_number=line_number,
                        status="review_required",
                        reasons=(exc.code,),
                        task_id=task_id,
                        source_record_id=source_record_id,
                        original_id=original_id,
                    )
                )

        ranked = sorted(
            accepted,
            key=lambda item: (self._rank(seed, item[1].task_id), item[1].task_id),
        )
        selected_pairs = ranked if max_tasks is None else ranked[:max_tasks]
        selected_ids = {task.task_id for _, task in selected_pairs}
        entries = [
            replace(
                entry,
                selected=entry.task_id in selected_ids,
            )
            for entry in entries
        ]
        selected_tasks = tuple(task for _, task in selected_pairs)
        return SourcePreflightResult(
            entries=tuple(entries),
            accepted_tasks=tuple(task for _, task in accepted),
            selected_tasks=selected_tasks,
            seed=seed,
            max_tasks=max_tasks,
        )
