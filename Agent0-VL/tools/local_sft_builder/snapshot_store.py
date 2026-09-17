"""Content-addressed snapshot persistence for a run directory.

``ImmutableSnapshot`` already hashes its own payload, and
``sha256(canonical_json_bytes(snapshot.payload)) == snapshot.snapshot_hash``
holds for both root snapshots and forks.  This store therefore writes the
canonical payload bytes verbatim, which makes every ``snap_<hash>.json`` file
tamper-evident: re-hashing the file must reproduce the name it is stored under.

Writing the same snapshot twice is idempotent, and a pre-existing file whose
content no longer matches its address is a hard error rather than a silent
overwrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json_bytes, sha256_bytes, sha256_file
from .snapshots import ImmutableSnapshot


SNAPSHOT_STORE_VERSION = "agent0vl.local_sft_builder.snapshot_store.v1"


class SnapshotStoreError(RuntimeError):
    """Raised when a stored snapshot does not match its content address."""


class SnapshotStore:
    """Persist snapshots as ``snap_<hash>.json`` plus an ``index.jsonl``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._written: dict[str, Path] = {}
        self._indexed: set[str] = set()

    def path_for(self, snapshot: ImmutableSnapshot) -> Path:
        return self.root / f"{snapshot.snapshot_id}.json"

    def write(self, snapshot: ImmutableSnapshot) -> Path:
        if not isinstance(snapshot, ImmutableSnapshot):
            raise TypeError("snapshot must be an ImmutableSnapshot")

        payload_bytes = canonical_json_bytes(snapshot.payload)
        if sha256_bytes(payload_bytes) != snapshot.snapshot_hash:
            raise SnapshotStoreError(
                "snapshot payload does not reproduce its own hash: "
                f"{snapshot.snapshot_id}"
            )

        path = self.path_for(snapshot)
        if path.exists():
            if sha256_file(path) != snapshot.snapshot_hash:
                raise SnapshotStoreError(
                    f"stored snapshot {path} does not match its content address"
                )
        else:
            # Raw bytes: canonical JSON is already LF-only, so the file hashes
            # to the snapshot hash on every platform.
            path.write_bytes(payload_bytes)

        self._written[snapshot.snapshot_id] = path
        self._index(snapshot, path)
        return path

    def write_all(
        self, snapshots: Iterable[ImmutableSnapshot]
    ) -> tuple[Path, ...]:
        return tuple(self.write(snapshot) for snapshot in snapshots)

    def read(self, snapshot_id: str) -> dict[str, Any]:
        path = self.root / f"{snapshot_id}.json"
        if not path.is_file():
            raise SnapshotStoreError(f"unknown snapshot: {snapshot_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def written_count(self) -> int:
        return len(self._written)

    @property
    def written_ids(self) -> tuple[str, ...]:
        return tuple(self._written)

    def _index(self, snapshot: ImmutableSnapshot, path: Path) -> None:
        if snapshot.snapshot_id in self._indexed:
            return
        payload = snapshot.payload
        row = {
            "snapshot_store_version": SNAPSHOT_STORE_VERSION,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_schema_version": snapshot.snapshot_schema_version,
            "parent_snapshot_id": snapshot.parent_snapshot_id,
            "parent_snapshot_hash": snapshot.parent_snapshot_hash,
            "before_step_index": snapshot.before_step_index,
            "state_hash": snapshot.state_hash,
            "branch_start_state_hash": snapshot.branch_start_state_hash,
            "message_count": len(payload["messages"]),
            "image_count": len(payload["original_images"]),
            "file": path.name,
        }
        with self.index_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        self._indexed.add(snapshot.snapshot_id)


__all__ = ["SNAPSHOT_STORE_VERSION", "SnapshotStore", "SnapshotStoreError"]
