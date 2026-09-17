"""Run manifest helpers with provenance and secret-safety checks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schema import DEDUP_KEY_VERSION, SCHEMA_VERSION, SNAPSHOT_SCHEMA_VERSION


MANIFEST_VERSION = "agent0vl.local_sft_builder.manifest.v1"
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password)")


def _assert_no_secret(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET_KEY_RE.search(str(key)) and item not in (None, "", False):
                raise ValueError(f"secret-like field is not allowed in {path}.{key}")
            _assert_no_secret(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secret(item, f"{path}[{index}]")


def build_manifest(
    *,
    run_id: str,
    base_sha: str,
    commit_sha: str | None = None,
    input_partitions: list[str] | tuple[str, ...] = (),
    output_root: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "dedup_key_version": DEDUP_KEY_VERSION,
        "run_id": run_id,
        "base_sha": base_sha,
        "commit_sha": commit_sha,
        "protocol_authority": "upstream_agent0_evaluator_runtime",
        "solver_protocol": "fenced_python",
        "verifier_repair_protocol": "json",
        "observation_wrapper": "[Code Execution Result]",
        "input_partitions": list(input_partitions),
        "output_root": output_root,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest.update(dict(extra))
    _assert_no_secret(manifest)
    return manifest


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    _assert_no_secret(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            dict(manifest),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
