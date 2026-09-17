from __future__ import annotations

import pytest

from tools.local_sft_builder.manifest import build_manifest, write_manifest


def test_manifest_records_protocol_versions_without_secrets(tmp_path) -> None:
    manifest = build_manifest(
        run_id="fixture-run",
        base_sha="a" * 64,
        commit_sha="b" * 40,
        input_partitions=["sft_stage1"],
    )
    assert manifest["solver_protocol"] == "fenced_python"
    assert manifest["dedup_key_version"] == "sft_exact_v1"
    output = tmp_path / "manifest.json"
    write_manifest(manifest, output)
    assert '"api_key"' not in output.read_text(encoding="utf-8")


def test_manifest_rejects_secret_like_values() -> None:
    with pytest.raises(ValueError):
        build_manifest(
            run_id="unsafe",
            base_sha="a" * 64,
            extra={"api_key": "should-not-be-serialized"},
        )
