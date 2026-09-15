"""Canonical Agent0-VL data contracts and stable identity helpers.

This module is deliberately dependency-light.  It is used by the Phase 0
fixtures as well as by the later acquisition, trajectory, and exporter
stages, so it must not import a model, tokenizer, dataset loader, or sandbox
implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence


SCHEMA_VERSION = "agent0vl.dataset.v1"
PROTOCOL_VERSION = "agent0vl.protocol.v1"
SCORER_VERSION = "agent0vl.scorer.v1"
RAW_MANIFEST_SCHEMA_VERSION = "agent0vl.raw-manifest.v1"
RAW_INDEX_SCHEMA_VERSION = "agent0vl.raw-index.v1"

MAX_STEPS = 8
MAX_REPAIRS = 2
REPAIR_THRESHOLD = 0.7
MAX_TOOL_STDOUT_CHARS = 512
MAX_TOOL_STDERR_CHARS = 512
MAX_OBSERVATION_TOKENS = 512


class SolverTurnType(str, Enum):
    REASONING = "reasoning"
    TOOL = "tool"
    FINAL = "final"
    INVALID = "invalid"


class TerminationReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS_WITHOUT_FINAL_ANSWER = "max_steps_without_final_answer"
    MAX_CONTEXT = "max_context"
    INVALID_SOLVER_TURN = "invalid_solver_turn"
    SANDBOX_FAILURE = "sandbox_failure"
    GENERATION_FAILURE = "generation_failure"
    MANUAL_ABORT = "manual_abort"


class ExecutionFailureKind(str, Enum):
    NONE = "none"
    USER_CODE = "user_code"
    INFRASTRUCTURE = "infrastructure"


class SplitDisposition(str, Enum):
    TRAIN_ALLOWED = "train_allowed"
    VALIDATION_ONLY = "validation_only"
    EVAL_ONLY = "eval_only"
    IGNORE = "ignore"


class LicenseStatus(str, Enum):
    VERIFIED = "verified"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


class DownloadMode(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL_REQUIRED = "manual_required"
    EXTERNAL_CREDENTIALS = "external_credentials"


def _jsonable(value: Any) -> Any:
    """Convert supported schema values into deterministic JSON values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    """Return the canonical UTF-8 JSON representation used for hashes."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_relative_text(value: str) -> str:
    """Canonicalize a relative path string without resolving filesystem links."""

    value = value.replace("\\", "/")
    if not value or value.startswith("/") or re.match(r"^[A-Za-z]:/", value):
        raise ValueError(f"path must be relative: {value!r}")

    raw_parts = value.split("/")
    if any(part in {".", ".."} for part in raw_parts):
        raise ValueError(f"path contains forbidden traversal component: {value!r}")

    normalized = unicodedata.normalize("NFC", PurePosixPath(*raw_parts).as_posix())
    if not normalized or normalized == ".":
        raise ValueError("path must not be empty")
    if normalized.startswith("../") or normalized == "..":
        raise ValueError(f"path escapes its root: {value!r}")
    return normalized


def canonical_relative_path(path: os.PathLike[str] | str, raw_root: os.PathLike[str] | str) -> str:
    """Return a portable relative path and reject external symlinks.

    The returned path is relative to ``raw_root``, uses ``/`` separators, and
    is NFC-normalized.  If the candidate exists through a symlink outside the
    raw root, it is rejected.  This keeps Windows and Linux acquisitions
    hash-compatible without allowing files outside the managed root into an
    inventory.
    """

    root = Path(raw_root).absolute()
    raw_value = os.fspath(path)
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside raw root: {path!r}") from exc

    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"symlink resolves outside raw root: {path!r}") from exc

    return _canonical_relative_text(PurePosixPath(*relative.parts).as_posix())


def canonical_inventory_path(path: str) -> str:
    """Canonicalize a path already stored relative to an acquisition root."""

    return _canonical_relative_text(path)


def _canonical_file_inventory(file_inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in file_inventory:
        path = canonical_inventory_path(str(entry["path"]))
        normalized.append(
            {
                "path": path,
                "size": int(entry["size"]),
                "sha256": str(entry["sha256"]).lower(),
            }
        )
    return sorted(normalized, key=lambda item: item["path"])


def content_fingerprint(file_inventory: Sequence[Mapping[str, Any]]) -> str:
    """Hash only the canonical raw file inventory.

    Revision, source identity, probe interpretation, and runtime metadata are
    intentionally excluded.  The same canonical inventory therefore always
    produces the same content fingerprint.
    """

    payload = {"files": _canonical_file_inventory(file_inventory)}
    return sha256_text(canonical_json(payload))


def _canonical_duplicate_groups(groups: Sequence[Sequence[str]]) -> list[list[str]]:
    canonical_groups = [sorted(canonical_inventory_path(path) for path in group) for group in groups]
    canonical_groups = [group for group in canonical_groups if len(group) >= 2]
    return sorted(canonical_groups, key=lambda group: group[0])


def probe_fingerprint(probe: "RawDatasetProbeResult | Mapping[str, Any]") -> str:
    """Hash the concrete result of a particular probe implementation."""

    if isinstance(probe, RawDatasetProbeResult):
        data = probe.to_dict()
    else:
        data = dict(probe)

    missing_images = sorted(
        canonical_inventory_path(path) for path in data.get("missing_images", [])
    )
    decode_failures = sorted(
        canonical_inventory_path(path) for path in data.get("decode_failures", [])
    )
    payload = {
        "sample_count": int(data.get("sample_count", 0)),
        "split_names": sorted(str(name) for name in data.get("split_names", [])),
        "splits": {
            str(name): int(count)
            for name, count in sorted(dict(data.get("splits", {})).items())
        },
        "image_count": int(data.get("image_count", 0)),
        "missing_images": missing_images,
        "decode_failures": decode_failures,
        "duplicate_groups": _canonical_duplicate_groups(
            data.get("duplicate_groups", [])
        ),
        "metadata_fields": sorted(
            str(name) for name in data.get("metadata_fields", [])
        ),
    }
    return sha256_text(canonical_json(payload))


def acquisition_id(
    dataset: str,
    source_fingerprint: str | None,
    resolved_revision: str | None,
    raw_content_fingerprint: str,
) -> str:
    payload = {
        "dataset": dataset,
        "source_fingerprint": source_fingerprint,
        "resolved_revision": resolved_revision,
        "content_fingerprint": raw_content_fingerprint,
    }
    return sha256_text(canonical_json(payload))


def verification_id(
    acquisition: str,
    probe: str,
    probe_schema_version: str,
    verification_schema_version: str,
    builder_code_revision: str,
    split_policy_fingerprint: str | None = None,
    license_review_fingerprint: str | None = None,
) -> str:
    payload = {
        "acquisition_id": acquisition,
        "probe_fingerprint": probe,
        "probe_schema_version": probe_schema_version,
        "verification_schema_version": verification_schema_version,
        "builder_code_revision": builder_code_revision,
        "split_policy_fingerprint": split_policy_fingerprint,
        "license_review_fingerprint": license_review_fingerprint,
    }
    return sha256_text(canonical_json(payload))


def has_images(images: Any) -> bool:
    """Robustly determine whether an image column contains image values."""

    if images is None:
        return False

    try:
        import numpy as np  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        np = None

    if np is not None and isinstance(images, np.ndarray):
        return images.size > 0
    if isinstance(images, (list, tuple)):
        return len(images) > 0
    return True


def tool_success_flags(call_execution_success: Sequence[bool]) -> tuple[bool, bool]:
    """Return the current runtime-compatible any/all tool success flags."""

    if not call_execution_success:
        return False, False
    values = [bool(value) for value in call_execution_success]
    return any(values), all(values)


@dataclass
class ImageAsset:
    asset_id: str
    sha256: str
    path: str | None = None
    bytes_data: bytes | None = None
    parent_asset_id: str | None = None
    created_step: int | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_bytes: bool = False) -> dict[str, Any]:
        result = {
            "asset_id": self.asset_id,
            "sha256": self.sha256,
            "path": self.path,
            "parent_asset_id": self.parent_asset_id,
            "created_step": self.created_step,
            "width": self.width,
            "height": self.height,
            "metadata": self.metadata,
        }
        if include_bytes:
            result["bytes"] = self.bytes_data
        return result


@dataclass
class ObservationEvent:
    step_index: int
    stdout: str = ""
    stderr: str = ""
    execution_valid: bool = True
    execution_failure_kind: ExecutionFailureKind = ExecutionFailureKind.NONE
    processed_image_asset_ids: list[str] = field(default_factory=list)
    observation_token_count: int | None = None

    def __post_init__(self) -> None:
        validate_step_index(self.step_index)


@dataclass
class GenerationBoundary:
    step_index: int
    role: str
    state_snapshot_id: str
    rendered_text_hash: str
    rendered_token_ids_hash: str | None = None
    image_asset_ids: list[str] = field(default_factory=list)
    multimodal_input_hash: str | None = None

    def __post_init__(self) -> None:
        validate_step_index(self.step_index)


@dataclass
class ConversationState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    original_images: list[ImageAsset] = field(default_factory=list)
    derived_images: list[ImageAsset] = field(default_factory=list)
    observation_events: list[ObservationEvent] = field(default_factory=list)
    boundaries: list[GenerationBoundary] = field(default_factory=list)

    def clone_for_rollout(self) -> "ConversationState":
        return copy.deepcopy(self)


@dataclass
class TaskRecord:
    task_id: str
    source: str
    question: str
    ground_truth: str
    answer_type: Literal["math", "exact", "choice", "list"] = "exact"
    source_revision: str | None = None
    license: str | None = None
    original_id: str | None = None
    official_split: str | None = None
    accepted_answers: list[str] = field(default_factory=list)
    capability_labels: list[str] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION


@dataclass
class TrajectoryStep:
    step_index: int
    solver_text: str
    turn_type: SolverTurnType = SolverTurnType.REASONING
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    call_execution_success: list[bool] = field(default_factory=list)
    step_tool_success_any: bool = False
    step_tool_success_all: bool = False
    verification: dict[str, Any] | None = None
    repair: dict[str, Any] | None = None
    regenerated_solver_text: str | None = None

    def __post_init__(self) -> None:
        validate_step_index(self.step_index)
        any_success, all_success = tool_success_flags(self.call_execution_success)
        self.step_tool_success_any = any_success
        self.step_tool_success_all = all_success


@dataclass
class TrajectoryRecord:
    trajectory_id: str
    task_id: str
    stage: Literal["stage1", "stage2"]
    generator: str
    sandbox_backend: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_answer: str | None = None
    confidence: float | None = None
    num_steps: int = 0
    num_repairs: int = 0
    termination_reason: TerminationReason | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    source_revision: str | None = None
    license: str | None = None
    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION

    @property
    def computed_num_steps(self) -> int:
        return len(self.steps)

    def validate(self) -> None:
        expected_indices = list(range(1, len(self.steps) + 1))
        actual_indices = [step.step_index for step in self.steps]
        if actual_indices != expected_indices:
            raise ValueError(
                f"trajectory steps must be contiguous 1-based indices: {actual_indices}"
            )
        if self.num_steps not in (0, len(self.steps)):
            raise ValueError("num_steps does not match the number of steps")
        if not 0 <= self.num_repairs <= MAX_REPAIRS:
            raise ValueError("num_repairs must be between 0 and MAX_REPAIRS")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("trajectory confidence must be in [0, 1]")


@dataclass
class SupervisionRecord:
    record_id: str
    record_type: Literal[
        "solver_positive",
        "verifier",
        "repair",
        "regeneration",
    ]
    source_trajectory_id: str
    source_step_index: int | None
    stage: Literal["stage1", "stage2"]
    messages: list[dict[str, Any]]
    images: list[ImageAsset] = field(default_factory=list)
    loss_target: str = "last_assistant_only"
    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION

    def validate_single_assistant_target(self) -> None:
        assistant_count = sum(
            message.get("role") == "assistant" for message in self.messages
        )
        if assistant_count != 1:
            raise ValueError("SFT supervision record must contain exactly one assistant")
        if not self.messages or self.messages[-1].get("role") != "assistant":
            raise ValueError("SFT target must be the final message")
        if self.source_step_index is not None:
            validate_step_index(self.source_step_index)


def flatten_sft_projection(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project a canonical transcript into a single-assistant SFT record.

    Historical assistant content is quoted inside deterministic user context
    labels.  The final assistant message remains the only trainable target.
    System messages are omitted because the official SFT entry point injects
    the system prompt separately.
    """

    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("canonical transcript must end with an assistant target")

    context_parts: list[str] = []
    labels = {
        "user": "[USER_CONTEXT]",
        "assistant": "[ASSISTANT_CONTEXT]",
        "tool": "[TOOL_CONTEXT]",
        "system": "[SYSTEM_CONTEXT]",
    }
    for message in messages[:-1]:
        role = str(message.get("role", "user"))
        if role == "system":
            continue
        content = str(message.get("content", ""))
        context_parts.append(f"{labels.get(role, '[CONTEXT]')}\n{content}")

    target = dict(messages[-1])
    result: list[dict[str, Any]] = []
    if context_parts:
        result.append({"role": "user", "content": "\n\n".join(context_parts)})
    result.append(target)
    return result


@dataclass
class RawDatasetProbeResult:
    sample_count: int = 0
    split_names: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    image_count: int = 0
    missing_images: list[str] = field(default_factory=list)
    decode_failures: list[str] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    metadata_fields: list[str] = field(default_factory=list)

    @property
    def decode_failure_count(self) -> int:
        return len(self.decode_failures)

    @property
    def duplicate_file_count(self) -> int:
        return sum(len(group) for group in self.duplicate_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "split_names": list(self.split_names),
            "splits": dict(self.splits),
            "image_count": self.image_count,
            "missing_images": list(self.missing_images),
            "decode_failures": list(self.decode_failures),
            "duplicate_groups": [list(group) for group in self.duplicate_groups],
            "metadata_fields": list(self.metadata_fields),
            "decode_failure_count": self.decode_failure_count,
            "duplicate_file_count": self.duplicate_file_count,
        }


@dataclass
class DatasetSpec:
    name: str
    source_type: Literal[
        "huggingface",
        "github",
        "official_url",
        "local",
    ]
    source_url: str | None = None
    repo_id: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    source_fingerprint: str | None = None
    license: str | None = None
    license_status: LicenseStatus = LicenseStatus.UNKNOWN
    citation: str | None = None
    expected_splits: list[str] = field(default_factory=list)
    split_policy: dict[str, SplitDisposition] = field(default_factory=dict)
    expected_files: list[str] = field(default_factory=list)
    expected_patterns: list[str] = field(default_factory=list)
    min_pattern_matches: dict[str, int] = field(default_factory=dict)
    fingerprint_include: list[str] | None = None
    fingerprint_exclude: list[str] = field(
        default_factory=lambda: [
            "manifest.json",
            "*.tmp",
            "*.lock",
            ".DS_Store",
            "cache/**",
            "logs/**",
        ]
    )
    usage: list[str] = field(default_factory=list)
    download_mode: DownloadMode = DownloadMode.AUTOMATIC
    allow_remote_code: bool = False
    remote_code_required: bool = False


@dataclass
class AcquisitionManifest:
    acquisition_id: str
    dataset: str
    source_fingerprint: str | None
    content_fingerprint: str
    raw_files: list[dict[str, Any]]
    requested_revision: str | None = None
    resolved_revision: str | None = None
    source_url: str | None = None
    repo_id: str | None = None
    license: str | None = None
    citation: str | None = None
    download_provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RAW_MANIFEST_SCHEMA_VERSION


@dataclass
class VerificationRecord:
    verification_id: str
    acquisition_id: str
    probe_fingerprint: str
    probe: RawDatasetProbeResult
    verification_status: VerificationStatus
    license_status: LicenseStatus
    split_policy_result: dict[str, SplitDisposition] = field(default_factory=dict)
    probe_schema_version: str = "agent0vl.probe.v1"
    verification_schema_version: str = "agent0vl.verification.v1"
    builder_code_revision: str = "unknown"
    verification_notes: list[str] = field(default_factory=list)


def validate_step_index(step_index: int) -> None:
    if not isinstance(step_index, int) or not 1 <= step_index <= MAX_STEPS:
        raise ValueError(f"step_index must be an integer in [1, {MAX_STEPS}]")


def clone_for_rollout(state: ConversationState) -> ConversationState:
    """Deep-copy all mutable ConversationState members for one rollout."""

    return copy.deepcopy(state)
