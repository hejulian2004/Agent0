"""Content-addressed immutable conversation snapshots."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import canonical_json_bytes, normalize_text, sha256_bytes, sha256_json
from .schema import SNAPSHOT_SCHEMA_VERSION


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_STATE_KEYS = (
    "messages",
    "original_images",
    "derived_images",
    "observation_events",
    "generation_boundaries",
    "sandbox_state_mode",
    "sandbox_state_hash",
)
_VALID_SANDBOX_MODES = frozenset({"stateless", "serialized", "reconstructed"})


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _logical_asset(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "image reference must be a mapping with asset_id and content_sha256"
        )

    asset_id = value.get("asset_id") or value.get("logical_asset_id")
    content_hash = value.get("content_sha256")

    if not asset_id or not isinstance(asset_id, str):
        raise ValueError("image reference requires a logical asset_id")
    asset_id = normalize_text(asset_id).replace("\\", "/")
    # Absolute paths would make the snapshot platform-specific. Callers must
    # provide a logical ID such as ``dataset/item-7/image-0`` instead.
    if asset_id.startswith("/") or re.match(r"^[A-Za-z]:/", asset_id):
        raise ValueError("absolute image paths are not valid snapshot asset IDs")
    if not isinstance(content_hash, str) or not _HASH_RE.fullmatch(content_hash):
        raise ValueError("content_sha256 must be a lowercase SHA256 hex digest")
    return {"asset_id": asset_id, "content_sha256": content_hash}


def _asset_list(values: Any) -> list[dict[str, str]]:
    return [_logical_asset(value) for value in (values or [])]


@dataclass(frozen=True)
class ImmutableSnapshot:
    """A snapshot whose hash is independent of audit metadata and filesystem paths."""

    snapshot_id: str
    snapshot_hash: str
    snapshot_schema_version: str
    parent_snapshot_id: str | None
    parent_snapshot_hash: str | None
    before_step_index: int
    branch_start_state_hash: str
    _frozen_payload: Any

    @classmethod
    def create(
        cls,
        *,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        original_images: list[Any] | tuple[Any, ...] = (),
        derived_images: list[Any] | tuple[Any, ...] = (),
        observation_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        generation_boundaries: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        sandbox_state_mode: str = "stateless",
        sandbox_state_hash: str | None = None,
        parent_snapshot: "ImmutableSnapshot | None" = None,
        before_step_index: int = 0,
        protocol_id: str = "agent0vl.runtime.fenced_python.v1",
        base_sha: str | None = None,
    ) -> "ImmutableSnapshot":
        if before_step_index < 0:
            raise ValueError("before_step_index must be non-negative")
        if sandbox_state_mode not in _VALID_SANDBOX_MODES:
            raise ValueError(f"invalid sandbox_state_mode: {sandbox_state_mode}")
        if sandbox_state_mode == "stateless" and sandbox_state_hash is not None:
            raise ValueError("stateless sandbox must have sandbox_state_hash=None")
        if sandbox_state_mode != "stateless" and sandbox_state_hash is None:
            raise ValueError(
                "serialized/reconstructed sandbox requires a deterministic state hash"
            )
        if sandbox_state_hash is not None and not _HASH_RE.fullmatch(sandbox_state_hash):
            raise ValueError("sandbox_state_hash must be a lowercase SHA256 hex digest")
        if base_sha is not None and not _HASH_RE.fullmatch(base_sha):
            raise ValueError("base_sha must be a lowercase SHA256 hex digest")

        parent_id = parent_snapshot.snapshot_id if parent_snapshot else None
        parent_hash = parent_snapshot.snapshot_hash if parent_snapshot else None
        payload = {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "parent_snapshot_hash": parent_hash,
            "before_step_index": before_step_index,
            "messages": copy.deepcopy(list(messages)),
            "original_images": _asset_list(original_images),
            "derived_images": _asset_list(derived_images),
            "observation_events": copy.deepcopy(list(observation_events)),
            "generation_boundaries": copy.deepcopy(list(generation_boundaries)),
            "sandbox_state_mode": sandbox_state_mode,
            "sandbox_state_hash": sandbox_state_hash,
            "protocol_id": normalize_text(protocol_id),
            "base_sha": base_sha,
        }
        state_payload = {key: payload[key] for key in _STATE_KEYS}
        branch_start_state_hash = (
            parent_snapshot.state_hash if parent_snapshot else sha256_json(state_payload)
        )
        payload["branch_start_state_hash"] = branch_start_state_hash
        canonical_bytes = canonical_json_bytes(payload)
        snapshot_hash = sha256_bytes(canonical_bytes)
        frozen = _freeze(payload)
        return cls(
            snapshot_id=f"snap_{snapshot_hash}",
            snapshot_hash=snapshot_hash,
            snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
            parent_snapshot_id=parent_id,
            parent_snapshot_hash=parent_hash,
            before_step_index=before_step_index,
            branch_start_state_hash=branch_start_state_hash,
            _frozen_payload=frozen,
        )

    @classmethod
    def fork(
        cls,
        parent: "ImmutableSnapshot",
        *,
        before_step_index: int,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        original_images: list[Any] | tuple[Any, ...] = (),
        derived_images: list[Any] | tuple[Any, ...] = (),
        observation_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        generation_boundaries: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        sandbox_state_mode: str = "stateless",
        sandbox_state_hash: str | None = None,
        protocol_id: str = "agent0vl.runtime.fenced_python.v1",
        base_sha: str | None = None,
    ) -> "ImmutableSnapshot":
        """Create a branch with a cryptographic reference to ``parent``."""

        return cls.create(
            messages=messages,
            original_images=original_images,
            derived_images=derived_images,
            observation_events=observation_events,
            generation_boundaries=generation_boundaries,
            sandbox_state_mode=sandbox_state_mode,
            sandbox_state_hash=sandbox_state_hash,
            parent_snapshot=parent,
            before_step_index=before_step_index,
            protocol_id=protocol_id,
            base_sha=base_sha,
        )

    @property
    def payload(self) -> dict[str, Any]:
        """Return a defensive copy; nested data cannot mutate the snapshot."""

        return _thaw(self._frozen_payload)

    @property
    def canonical_payload(self) -> dict[str, Any]:
        return self.payload

    @property
    def state_payload(self) -> dict[str, Any]:
        payload = self.payload
        return {key: payload[key] for key in _STATE_KEYS}

    @property
    def state_hash(self) -> str:
        return sha256_json(self.state_payload)

    def branch_start_matches_parent(self, parent: "ImmutableSnapshot") -> bool:
        return (
            self.parent_snapshot_id == parent.snapshot_id
            and self.parent_snapshot_hash == parent.snapshot_hash
            and self.branch_start_state_hash == parent.state_hash
        )
