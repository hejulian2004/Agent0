"""End-to-end tests for the ``--generate`` run directory.

The Teacher is a local ``http.server`` stub, the sandbox is the real frozen
upstream CPU sandbox, and the source is a synthetic single-row JSONL.  No
dataset, network endpoint or model is touched.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from tools.local_sft_builder.answer_check import (
    ANSWER_CHECK_METHOD,
    EVIDENCE_SOURCE,
)
from tools.local_sft_builder.canonical import sha256_file
from tools.local_sft_builder.run_manifest import PHASE2B_MANIFEST_VERSION
from tools.local_sft_builder.run_real_builder import (
    EXPECTED_BASE_SHA,
    _parser,
    _resolve_solver_prompt_addendum,
    main,
)
from tools.local_sft_builder.source_normalize import GROUND_TRUTH_ORIGIN
from tools.local_sft_builder.trajectory_builder import SOLVER_PROMPT_ADDENDUM


CREDENTIAL_ENV = "AGENT0_TEACHER_API_KEY"
CREDENTIAL_VALUE = "stub-credential-do-not-publish"


def _minimal_generate_argv() -> list[str]:
    return [
        "--generate",
        "--source-path",
        "normalized.jsonl",
        "--stage",
        "stage1",
        "--output-run-dir",
        "run",
        "--max-tasks",
        "1",
        "--base-sha",
        EXPECTED_BASE_SHA,
        "--phase1-freeze-sha",
        EXPECTED_BASE_SHA,
    ]


def test_default_max_tokens_leaves_room_for_think_and_answer() -> None:
    """The default must not truncate the response before the answer appears.

    Measured on the real Teacher: a 1024-token budget is consumed by the
    ``<think>`` block before the final answer or a code block is ever emitted,
    which failed every task in a 5-task smoke run.  2048 is the legacy builder's
    default, and the production scripts pass 4096 explicitly.
    """

    args = _parser().parse_args(_minimal_generate_argv())

    assert args.max_tokens >= 2048


def test_solver_prompt_addendum_is_on_by_default_and_can_be_disabled() -> None:
    """The tool-use addendum is the default; ``--no-...`` reproduces the old run."""

    default_args = _parser().parse_args(_minimal_generate_argv())
    assert _resolve_solver_prompt_addendum(default_args) == SOLVER_PROMPT_ADDENDUM

    disabled = _parser().parse_args(
        _minimal_generate_argv() + ["--no-solver-prompt-addendum"]
    )
    assert _resolve_solver_prompt_addendum(disabled) is None

FINAL_TURN = (
    "<think>The tallest bar reaches 9.</think>\n"
    "CONFIDENCE: 0.9\n"
    "FINAL_ANSWER: 9"
)
CODE_TURN = "<think>I need to compute this.</think>\n```python\nprint(9)\n```"

IMAGE_RELATIVE = "cauldron/clevr/images/clevr_00011395.png"
QUESTION = "<image>\nQuestion: What is the value of the largest bar?"

EXPECTED_FILES = {
    "manifest.json",
    "source_index.jsonl",
    "rejected.jsonl",
    "teacher_requests.sqlite3",
    "snapshots",
    "trajectories.jsonl",
    "audit_records.jsonl",
    "validation_decisions.jsonl",
    "export_candidates.jsonl",
    "stage1.jsonl",
    "stage2.jsonl",
    "final_dedup.jsonl",
}

# Phase 2B-Lite dropped the ledger-finalization step, so the ledger is no
# longer converted out of WAL mode and ``teacher_requests.sqlite3`` may or may
# not be accompanied by ``-wal`` / ``-shm`` sidecars. Whether they survive the
# process is not an acceptance condition either way (plan 4.B), so the checks
# below are blind to them instead of pinning an exact file count.
SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        state = self.server.stub_state  # type: ignore[attr-defined]
        state["requests"].append(
            {
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "payload": json.loads(raw.decode("utf-8")),
            }
        )
        contents = state["contents"]
        index = min(state["index"], len(contents) - 1)
        state["index"] += 1
        body = json.dumps(
            {
                "id": "chatcmpl-stub",
                "model": "stub-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": contents[index]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # silence the stub
        return


class StubTeacher:
    """A local OpenAI-compatible endpoint replaying scripted contents."""

    def __init__(self, contents: list[str]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._server.stub_state = {  # type: ignore[attr-defined]
            "requests": self.requests,
            "contents": list(contents),
            "index": 0,
        }
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "StubTeacher":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _write_source(tmp_path: Path, *, ground_truth: str = "9") -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_path = image_root / IMAGE_RELATIVE
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"phase2b smoke image")

    row = {
        "task_id": "mulberry-stage1-000001",
        "source_record_id": "mulberry:deadbeef",
        "original_id": "mulberry-260494",
        "source_dataset": "mulberry",
        "source_revision": "486ed06f81be192a14ca7f71ea54d81db1ab084e",
        "split": "train",
        "usage_partition": "sft_stage1",
        "question": QUESTION,
        "ground_truth": ground_truth,
        "images": [IMAGE_RELATIVE],
        "image_refs": [
            {
                "asset_id": "mulberry/mulberry-260494/image-0",
                "content_sha256": sha256_file(image_path),
            }
        ],
    }
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return source_path, image_root


def _args(
    source_path: Path,
    image_root: Path,
    output_dir: Path,
    base_url: str,
    *extra: str,
) -> list[str]:
    return [
        "--generate",
        "--source-path",
        str(source_path),
        "--stage",
        "stage1",
        "--output-run-dir",
        str(output_dir),
        "--image-root",
        str(image_root),
        "--image-root-config",
        "mulberry_images_v1",
        "--max-tasks",
        "1",
        "--seed",
        "0",
        "--base-sha",
        EXPECTED_BASE_SHA,
        "--phase1-freeze-sha",
        "HEAD",
        "--teacher-base-url",
        base_url,
        "--teacher-model",
        "stub-model",
        *extra,
    ]


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_generate_writes_the_full_run_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(CREDENTIAL_ENV, CREDENTIAL_VALUE)
    source_path, image_root = _write_source(tmp_path)
    output_dir = tmp_path / "run"

    with StubTeacher([CODE_TURN, FINAL_TURN]) as teacher:
        exit_code = main(_args(source_path, image_root, output_dir, teacher.base_url))
        requests = list(teacher.requests)

    assert exit_code == 0
    produced_names = {
        path.name
        for path in output_dir.iterdir()
        if not path.name.endswith(SQLITE_SIDECAR_SUFFIXES)
    }
    assert produced_names == EXPECTED_FILES

    # --- the exported row -------------------------------------------------
    rows = _jsonl(output_dir / "final_dedup.jsonl")
    assert len(rows) == 1
    assert set(rows[0]) == {"messages", "images"}
    # Portable path, never the resolved machine path.
    assert rows[0]["images"] == [IMAGE_RELATIVE]
    assert [message["role"] for message in rows[0]["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert rows[0]["messages"][2]["content"].startswith("\n[Code Execution Result]\n")
    assert "Output: 9" in rows[0]["messages"][2]["content"]
    assert rows[0]["messages"][3]["content"] == FINAL_TURN

    assert _jsonl(output_dir / "stage1.jsonl") == rows
    assert _jsonl(output_dir / "stage2.jsonl") == []

    # --- the Teacher received the source Solver protocol and the image -----
    assert len(requests) == 2
    assert requests[0]["headers"]["authorization"] == f"Bearer {CREDENTIAL_VALUE}"
    first_messages = requests[0]["payload"]["messages"]
    assert first_messages[0]["role"] == "system"
    assert "```python" in first_messages[0]["content"]
    first_blocks = first_messages[1]["content"]
    assert isinstance(first_blocks, list)
    assert first_blocks[0]["type"] == "image_url"
    assert first_blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Question: What is the value of the largest bar?" in first_blocks[1]["text"]
    # The exported user turn is the source question, not a local scaffold.
    assert "FINAL_ANSWER:" not in first_blocks[1]["text"]
    # The second request resends the accumulated context: system + 3 turns.
    assert len(requests[1]["payload"]["messages"]) == 4

    # --- snapshots are content addressed ----------------------------------
    index_rows = _jsonl(output_dir / "snapshots" / "index.jsonl")
    assert len(index_rows) == 1
    snapshot_file = output_dir / "snapshots" / index_rows[0]["file"]
    assert snapshot_file.name == f"{index_rows[0]['snapshot_id']}.json"
    assert sha256_file(snapshot_file) == index_rows[0]["snapshot_hash"]
    assert index_rows[0]["parent_snapshot_id"] is None
    assert index_rows[0]["image_count"] == 1

    # --- trajectories, audit and decisions --------------------------------
    assert len(_jsonl(output_dir / "trajectories.jsonl")) == 1
    audit_kinds = {
        record["record_kind"] for record in _jsonl(output_dir / "audit_records.jsonl")
    }
    assert {
        "task_analysis",
        "natural_rollout",
        "image_provenance",
        "reference_answer_check",
        "validation_decision",
    } <= audit_kinds

    answer_audit = next(
        record
        for record in _jsonl(output_dir / "audit_records.jsonl")
        if record["record_kind"] == "reference_answer_check"
    )
    # R1: the weak Mulberry label backs a *reference match*, never independent
    # verification, so the audit must say so explicitly.
    assert answer_audit["payload"]["answer_check_method"] == ANSWER_CHECK_METHOD
    assert answer_audit["payload"]["ground_truth_origin"] == GROUND_TRUTH_ORIGIN
    assert answer_audit["payload"]["independently_verified"] is False
    assert answer_audit["payload"]["reference_label_is_weak"] is True
    assert answer_audit["payload"]["consistency"] == "match"

    decisions = _jsonl(output_dir / "validation_decisions.jsonl")
    assert [decision["status"] for decision in decisions] == ["accepted"]
    # This is the *frozen* Validator's own vocabulary, which R1 forbids us from
    # changing. It is deliberately not the builder's evidence type: the honest
    # `reference_answer_match` provenance lives in the audit record and the
    # manifest, so a reader of this file alone must not read it as independent
    # verification.
    assert decisions[0]["evidence_sources"] == ["deterministic_rule"]

    candidates = _jsonl(output_dir / "export_candidates.jsonl")
    assert len(candidates) == 1
    assert candidates[0]["images"] == [IMAGE_RELATIVE]

    # --- ledger -----------------------------------------------------------
    assert _jsonl(output_dir / "rejected.jsonl") == []
    source_entries = _jsonl(output_dir / "source_index.jsonl")
    assert [entry["status"] for entry in source_entries] == ["accepted"]

    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["phase2b_manifest_version"] == PHASE2B_MANIFEST_VERSION
    assert manifest["teacher_enabled"] is True
    assert manifest["training_enabled"] is False
    assert manifest["image_path_form"] == "relative_to_image_root"
    assert manifest["total_source_rows"] == 1
    assert manifest["selected_after_sampling"] == 1
    assert manifest["natural_success"] == 1
    assert manifest["natural_failure"] == 0
    assert manifest["export_candidates"] == 1
    assert manifest["dedup_dropped"] == 0
    assert manifest["final_sft_rows"] == 1
    assert manifest["snapshot_count"] == 1
    assert manifest["trajectory_count"] == 1
    assert manifest["teacher_request_counts"] == {
        "consumed_slot": 2,
        "pending": 0,
        "successful": 2,
        "timeout": 0,
        "parse_failed": 0,
        "other_failed": 0,
    }

    # Phase 2B-Lite lowest guarantee: once ``--generate`` has returned, the
    # ledger must still be re-openable and must report the same request count
    # the manifest published. It may well still be in WAL mode -- that is
    # accepted, and the sidecars are deliberately not asserted on. Note the
    # explicit ``close()``: ``with sqlite3.connect(...)`` only ends the
    # transaction and never closes the handle.
    ledger = sqlite3.connect(str(output_dir / "teacher_requests.sqlite3"))
    try:
        recorded_requests = ledger.execute(
            "SELECT COUNT(*) FROM teacher_requests"
        ).fetchone()[0]
    finally:
        ledger.close()
    assert recorded_requests == manifest["teacher_request_counts"]["consumed_slot"]

    assert manifest["teacher"]["teacher_credential_configured"] is True
    assert manifest["teacher"]["teacher_credential_env"] == CREDENTIAL_ENV
    # The manifest guard rejects "token", so the limit is renamed.
    # 2048, not 1024: a 1024-token budget is consumed by the ``<think>`` block
    # before the answer or a code block is emitted (measured on the real Teacher).
    assert manifest["teacher"]["sampling"]["max_output"] == 2048
    assert "max_tokens" not in manifest["teacher"]["sampling"]
    assert manifest["sandbox"]["sandbox_backend"] == "upstream_agent0_vl_sandbox"
    # R1 provenance, published next to the frozen Validator's own vocabulary.
    assert manifest["builder"]["answer_check_method"] == ANSWER_CHECK_METHOD
    assert manifest["builder"]["answer_check_evidence_source"] == EVIDENCE_SOURCE
    assert manifest["builder"]["independently_verified"] is False
    assert manifest["builder"]["ground_truth_origin"] == GROUND_TRUTH_ORIGIN
    assert manifest["builder"]["task_analysis_enabled"] is False
    assert manifest["builder"]["step_validation"] == "not_performed_phase2b_p0"
    # The Solver protocol travels as a system message on the request and is
    # deliberately absent from the exported rows.
    assert manifest["builder"]["solver_system_prompt_role"] == "system"
    assert manifest["builder"]["solver_system_prompt_exported"] is False
    assert isinstance(manifest["builder"]["solver_system_prompt_sha256"], str)
    assert len(manifest["builder"]["solver_system_prompt_sha256"]) == 64

    # --- the credential never reaches any run artifact --------------------
    for path in output_dir.rglob("*"):
        if path.is_file() and path.suffix != ".sqlite3":
            assert CREDENTIAL_VALUE not in path.read_text(
                encoding="utf-8", errors="replace"
            ), path


def test_generate_requires_the_credential_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(CREDENTIAL_ENV, raising=False)
    source_path, image_root = _write_source(tmp_path)
    output_dir = tmp_path / "run"

    with StubTeacher([FINAL_TURN]) as teacher:
        exit_code = main(_args(source_path, image_root, output_dir, teacher.base_url))
        requests = list(teacher.requests)

    assert exit_code == 2
    assert requests == []
    assert not output_dir.exists()


def test_generate_refuses_to_overwrite_an_existing_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(CREDENTIAL_ENV, CREDENTIAL_VALUE)
    source_path, image_root = _write_source(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    with StubTeacher([FINAL_TURN]) as teacher:
        exit_code = main(_args(source_path, image_root, output_dir, teacher.base_url))

    assert exit_code == 2
    assert list(output_dir.iterdir()) == []


def test_generate_and_dry_run_are_mutually_exclusive(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(CREDENTIAL_ENV, CREDENTIAL_VALUE)
    source_path, image_root = _write_source(tmp_path)
    output_dir = tmp_path / "run"

    with pytest.raises(SystemExit) as both:
        main(
            [
                *_args(source_path, image_root, output_dir, "http://127.0.0.1:1"),
                "--dry-run",
            ]
        )
    assert both.value.code == 2
    assert not output_dir.exists()


def test_a_missing_mode_is_rejected(tmp_path) -> None:
    with pytest.raises(SystemExit) as missing:
        main(["--source-path", "x.jsonl", "--stage", "stage1"])
    assert missing.value.code == 2


def test_mismatching_reference_answer_yields_no_export_rows(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(CREDENTIAL_ENV, CREDENTIAL_VALUE)
    source_path, image_root = _write_source(tmp_path, ground_truth="112")
    output_dir = tmp_path / "run"

    with StubTeacher([FINAL_TURN]) as teacher:
        exit_code = main(_args(source_path, image_root, output_dir, teacher.base_url))

    assert exit_code == 0
    assert _jsonl(output_dir / "final_dedup.jsonl") == []
    assert _jsonl(output_dir / "export_candidates.jsonl") == []
    decisions = _jsonl(output_dir / "validation_decisions.jsonl")
    assert [decision["status"] for decision in decisions] == ["review_required"]
    assert "final_answer_not_independently_validated" in decisions[0]["reasons"]

    manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["natural_success"] == 1
    assert manifest["export_candidates"] == 0
    assert manifest["final_sft_rows"] == 0
