"""Phase 2A run-manifest construction and input fingerprinting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import normalize_text, sha256_file, sha256_json
from .manifest import build_manifest, write_manifest


PHASE2A_MANIFEST_VERSION = "agent0vl.local_sft_builder.phase2a.manifest.v1"


def _resolved_file_map(paths: Sequence[str | Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        result[str(path)] = sha256_file(path)
    return result


def build_phase2a_manifest(
    *,
    run_id: str,
    base_sha: str,
    phase1_freeze_sha: str,
    builder_commit_sha: str,
    source_paths: Sequence[str | Path],
    forbidden_paths: Sequence[str | Path],
    image_root_config: str | None,
    image_root_resolved_path: str | Path | None,
    stage: str,
    seed: int,
    max_tasks: int | None,
    total_source_rows: int,
    accepted_before_sampling: int,
    selected_after_sampling: int,
    output_root: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(phase1_freeze_sha, str) or not phase1_freeze_sha.strip():
        raise ValueError("phase1_freeze_sha must be a non-empty commit SHA")
    if not isinstance(builder_commit_sha, str) or not builder_commit_sha.strip():
        raise ValueError("builder_commit_sha must be a non-empty commit SHA")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(total_source_rows, int) or total_source_rows < 0:
        raise ValueError("total_source_rows must be a non-negative integer")
    if not isinstance(accepted_before_sampling, int) or accepted_before_sampling < 0:
        raise ValueError("accepted_before_sampling must be a non-negative integer")
    if not isinstance(selected_after_sampling, int) or selected_after_sampling < 0:
        raise ValueError("selected_after_sampling must be a non-negative integer")

    source_file_sha256 = _resolved_file_map(source_paths)
    forbidden_file_sha256 = _resolved_file_map(forbidden_paths)
    resolved_root = (
        str(Path(image_root_resolved_path).expanduser().resolve())
        if image_root_resolved_path is not None
        else None
    )
    normalized_image_root_config = (
        normalize_text(image_root_config).strip().replace("\\", "/")
        if image_root_config is not None
        else None
    )
    image_root_identity = (
        sha256_json({"image_root_config": normalized_image_root_config})
        if normalized_image_root_config is not None
        else None
    )

    manifest_extra: dict[str, Any] = {
        "phase2a_manifest_version": PHASE2A_MANIFEST_VERSION,
        "phase1_freeze_sha": phase1_freeze_sha,
        "builder_commit_sha": builder_commit_sha,
        "source_file_sha256": source_file_sha256,
        "forbidden_file_sha256": forbidden_file_sha256,
        "image_root_config": normalized_image_root_config,
        "image_root_resolved_path": resolved_root,
        "image_root_identity": image_root_identity,
        "stage": stage,
        "seed": seed,
        "max_tasks": max_tasks,
        "total_source_rows": total_source_rows,
        "accepted_before_sampling": accepted_before_sampling,
        "selected_after_sampling": selected_after_sampling,
        "teacher_enabled": False,
        "snapshot_enabled": False,
        "trajectory_enabled": False,
        "sft_export_enabled": False,
        "training_enabled": False,
    }
    if extra:
        manifest_extra.update(dict(extra))
    return build_manifest(
        run_id=run_id,
        base_sha=base_sha,
        commit_sha=builder_commit_sha,
        input_partitions=(stage,),
        output_root=str(Path(output_root).expanduser().resolve()),
        extra=manifest_extra,
    )


__all__ = ["PHASE2A_MANIFEST_VERSION", "build_phase2a_manifest", "write_manifest"]
