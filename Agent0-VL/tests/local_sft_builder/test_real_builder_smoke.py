from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tools.local_sft_builder.canonical import sha256_file
from tools.local_sft_builder.run_real_builder import main
from tools.local_sft_builder.run_manifest import build_phase2a_manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row(*, image_root: Path) -> dict:
    image_path = image_root / "image.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"phase2a smoke image")
    return {
        "task_id": "smoke-task",
        "source_record_id": "smoke-record",
        "original_id": "smoke-original",
        "source_dataset": "fixture",
        "source_revision": "fixture-v1",
        "split": "train",
        "usage_partition": "sft_stage1",
        "question": "<image>\nWhat is shown?",
        "ground_truth": "",
        "images": ["image.png"],
        "image_refs": [
            {
                "asset_id": "fixture/smoke/image-0",
                "content_sha256": sha256_file(image_path),
            }
        ],
    }


def test_phase2a_dry_run_writes_only_preflight_outputs(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [_row(image_root=image_root)])
    output_dir = tmp_path / "run"

    exit_code = main(
        [
            "--source-path",
            str(source_path),
            "--stage",
            "stage1",
            "--output-run-dir",
            str(output_dir),
            "--image-root",
            str(image_root),
            "--image-root-config",
            "fixture-images-v1",
            "--max-tasks",
            "1",
            "--seed",
            "0",
            "--base-sha",
            "HEAD",
            "--phase1-freeze-sha",
            "HEAD",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert {
        path.name for path in output_dir.iterdir()
    } == {
        "manifest.json",
        "source_index.jsonl",
        "rejected.jsonl",
        "teacher_requests.sqlite3",
    }

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase2a_manifest_version"].endswith("manifest.v1")
    assert manifest["base_sha"] == manifest["builder_commit_sha"]
    assert manifest["phase1_freeze_sha"] == manifest["builder_commit_sha"]
    assert manifest["source_file_sha256"][str(source_path.resolve())] == sha256_file(source_path)
    assert manifest["image_root_config"] == "fixture-images-v1"
    assert manifest["image_root_resolved_path"] == str(image_root.resolve())
    assert manifest["total_source_rows"] == 1
    assert manifest["accepted_before_sampling"] == 1
    assert manifest["selected_after_sampling"] == 1
    assert manifest["seed"] == 0
    assert manifest["max_tasks"] == 1
    assert manifest["teacher_enabled"] is False
    assert manifest["snapshot_enabled"] is False
    assert manifest["trajectory_enabled"] is False

    source_entries = [
        json.loads(line)
        for line in (output_dir / "source_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(source_entries) == 1
    assert source_entries[0]["status"] == "accepted"
    assert source_entries[0]["selected"] is True
    assert source_entries[0]["source_content_hash_origin"] == "derived"
    assert (output_dir / "rejected.jsonl").read_text(encoding="utf-8") == ""

    with sqlite3.connect(output_dir / "teacher_requests.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM teacher_requests").fetchone()[0] == 0


def test_phase2a_dry_run_rejects_output_overwrite(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(source_path, [])
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    exit_code = main(
        [
            "--source-path",
            str(source_path),
            "--stage",
            "stage1",
            "--output-run-dir",
            str(output_dir),
            "--max-tasks",
            "1",
            "--phase1-freeze-sha",
            "HEAD",
            "--dry-run",
        ]
    )

    assert exit_code == 2


def test_manifest_image_root_identity_excludes_machine_path(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    first_root = tmp_path / "first-images"
    second_root = tmp_path / "second-images"
    first_root.mkdir()
    second_root.mkdir()

    common = {
        "base_sha": "a" * 64,
        "phase1_freeze_sha": "b" * 40,
        "builder_commit_sha": "c" * 40,
        "source_paths": (source_path,),
        "forbidden_paths": (),
        "image_root_config": r"fixture\images",
        "stage": "sft_stage1",
        "seed": 0,
        "max_tasks": 1,
        "total_source_rows": 0,
        "accepted_before_sampling": 0,
        "selected_after_sampling": 0,
    }
    first = build_phase2a_manifest(
        run_id="first",
        image_root_resolved_path=first_root,
        output_root=tmp_path / "first-run",
        **common,
    )
    second = build_phase2a_manifest(
        run_id="second",
        image_root_resolved_path=second_root,
        output_root=tmp_path / "second-run",
        **common,
    )

    assert first["image_root_config"] == "fixture/images"
    assert first["image_root_identity"] == second["image_root_identity"]
    assert first["image_root_resolved_path"] != second["image_root_resolved_path"]
