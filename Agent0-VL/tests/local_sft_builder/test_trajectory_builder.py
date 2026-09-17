"""Tests for the real natural rollout builder (Phase 2B-P0).

Everything here is synthetic: a scripted Teacher, a recording sandbox and an
in-memory task.  No dataset, endpoint or real sandbox is touched.
"""

from __future__ import annotations

import json

import pytest

from tools.local_sft_builder.answer_check import MATCH, MISMATCH
from tools.local_sft_builder.budget import TeacherRequestBudget
from tools.local_sft_builder.fake_builder import SourceTask
from tools.local_sft_builder.manifest import build_manifest
from tools.local_sft_builder.projector import Projector
from tools.local_sft_builder.runtime import SandboxResult, SourceRuntimeAdapter
from tools.local_sft_builder.teacher_backend import (
    TeacherBackendError,
    TeacherResponse,
)
from tools.local_sft_builder.trajectory_builder import (
    FAILURE_JSON_TOOL_CALL,
    FAILURE_MAX_TURNS,
    FAILURE_NO_TERMINAL,
    GROUND_TRUTH_ORIGIN,
    SOLVER_PROMPT_SCAFFOLD,
    SOLVER_PROMPT_VERSION,
    STEP_VALIDATION_NOTE,
    TERMINAL_FINAL_ANSWER,
    RealTrajectoryBuilder,
    build_solver_user_content,
)
from tools.local_sft_builder.validator import validate_trajectory_for_export


IMAGE_SHA = "b" * 64
# The physical path is what the runtime and the Teacher need; the exported row
# must carry the image-root-relative form instead.
IMAGE_PHYSICAL = (
    "/mnt/d/Agent0-dev/Agent0-VL/data/raw/.staging/mulberry-proxy/"
    "cauldron/clevr/images/clevr_00011395.png"
)
IMAGE_RELATIVE = "cauldron/clevr/images/clevr_00011395.png"
FINAL_TURN = (
    "<think>The tallest bar reaches 9.</think>\n"
    "CONFIDENCE: 0.9\n"
    "FINAL_ANSWER: 9"
)
CODE_TURN = "<think>I need to compute this.</think>\n```python\nprint(9)\n```"


class RecordingSandbox:
    """Deterministic sandbox stub that records every snippet it ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, code: str) -> SandboxResult:
        self.calls.append(code)
        return SandboxResult(
            status="success",
            stdout=f"value={code.strip()}\n",
            stderr="",
        )


class StubTeacher:
    """Scripted Teacher with an observable request log."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def generate_from_payload(self, payload: dict) -> TeacherResponse:
        self.requests.append(dict(payload))
        if not self._responses:
            raise TeacherBackendError("no scripted response left")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return TeacherResponse(text=str(item), model="stub", finish_reason="stop")


def _task(**overrides: object) -> SourceTask:
    payload: dict[str, object] = {
        "task_id": "mulberry-stage1-000001",
        "source_record_id": "mulberry:deadbeef",
        "original_id": "mulberry-260494",
        "source_dataset": "mulberry",
        "stage": "sft_stage1",
        "usage_partition": "sft_stage1",
        "question": "<image>Question: What is the value of the largest bar?",
        "images": (IMAGE_PHYSICAL,),
        "image_refs": (
            {"asset_id": "mulberry/item-1/image-0", "content_sha256": IMAGE_SHA},
        ),
        "image_relatives": (IMAGE_RELATIVE,),
        "ground_truth": "9",
    }
    payload.update(overrides)
    return SourceTask(**payload)  # type: ignore[arg-type]


def _builder(
    tmp_path,
    responses: list[object],
    **kwargs: object,
) -> tuple[RealTrajectoryBuilder, StubTeacher, RecordingSandbox]:
    teacher = StubTeacher(responses)
    sandbox = RecordingSandbox()
    builder = RealTrajectoryBuilder(
        backend=teacher,
        budget_db=str(tmp_path / "teacher_requests.sqlite3"),
        sandbox_runtime=SourceRuntimeAdapter(sandbox=sandbox),
        **kwargs,  # type: ignore[arg-type]
    )
    return builder, teacher, sandbox


def _audit(result, kind: str) -> dict:
    return next(
        record.payload for record in result.audit_records if record.record_kind == kind
    )


# --------------------------------------------------------------------------
# Prompt scaffolding
# --------------------------------------------------------------------------


def test_scaffold_embeds_the_source_question_verbatim() -> None:
    question = "<image>Question: What is the value of the largest bar?"

    content = build_solver_user_content(question)

    assert question in content
    assert content.count("<image>") == question.count("<image>") == 1
    assert content.startswith(SOLVER_PROMPT_SCAFFOLD)
    for marker in ("<think>", "```python", "CONFIDENCE:", "FINAL_ANSWER:"):
        assert marker in content


def test_scaffold_rejects_an_empty_question() -> None:
    with pytest.raises(ValueError):
        build_solver_user_content("   ")


def test_scaffold_preserves_multiple_image_placeholders() -> None:
    question = "<image>Compare <image> and answer."

    content = build_solver_user_content(question)

    assert content.count("<image>") == 2


# --------------------------------------------------------------------------
# Termination contract
# --------------------------------------------------------------------------


def test_final_answer_terminates_without_execution(tmp_path) -> None:
    builder, _teacher, sandbox = _builder(tmp_path, [FINAL_TURN])

    result = builder.build_task(_task())

    assert sandbox.calls == []
    assert result.rollout is not None
    assert result.rollout.completed is True
    assert result.rollout.terminal_reason == TERMINAL_FINAL_ANSWER
    assert result.rollout.turn_count == 1
    assert result.rollout.executed_step_count == 0


def test_final_answer_wins_over_a_code_block_in_the_same_response(tmp_path) -> None:
    builder, _teacher, sandbox = _builder(tmp_path, [f"{FINAL_TURN}\n{CODE_TURN}"])

    result = builder.build_task(_task())

    assert sandbox.calls == []
    assert result.rollout.terminal_reason == TERMINAL_FINAL_ANSWER


def test_response_with_neither_answer_nor_code_is_a_hard_failure(tmp_path) -> None:
    builder, _teacher, sandbox = _builder(
        tmp_path, ["<think>I am still thinking about it.</think>"]
    )

    result = builder.build_task(_task())

    assert sandbox.calls == []
    assert result.rollout.completed is False
    assert result.rollout.terminal_reason == FAILURE_NO_TERMINAL
    assert (
        result.rollout.terminal_reason
        == "solver_response_has_neither_final_answer_nor_executable_code"
    )
    assert result.trajectories == ()
    assert result.candidates == ()


def test_json_tool_call_response_is_rejected(tmp_path) -> None:
    payload = json.dumps({"tool_name": "PythonExec", "tool_input": "1 + 1"})
    builder, _teacher, sandbox = _builder(tmp_path, [payload])

    result = builder.build_task(_task())

    assert sandbox.calls == []
    assert result.rollout.terminal_reason == FAILURE_JSON_TOOL_CALL


def test_exceeding_max_rollout_turns_fails(tmp_path) -> None:
    builder, _teacher, sandbox = _builder(
        tmp_path, [CODE_TURN, CODE_TURN, CODE_TURN], max_rollout_turns=2
    )

    result = builder.build_task(_task())

    assert len(sandbox.calls) == 2
    assert result.rollout.terminal_reason == FAILURE_MAX_TURNS
    assert result.budget_stats.consumed_slot == 2


# --------------------------------------------------------------------------
# Multi-turn execution
# --------------------------------------------------------------------------


def test_multi_turn_rollout_executes_and_reinjects_the_observation(tmp_path) -> None:
    builder, teacher, sandbox = _builder(tmp_path, [CODE_TURN, FINAL_TURN])

    result = builder.build_task(_task())

    assert sandbox.calls == ["print(9)"]
    assert result.rollout.turn_count == 2
    assert result.rollout.executed_step_count == 1

    messages = result.rollout.messages
    assert [item["role"] for item in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[2]["content"].startswith("\n[Code Execution Result]\n")
    assert "Output: value=print(9)\n" in messages[2]["content"]
    assert messages[3]["content"] == FINAL_TURN

    # The stateless Teacher must be re-sent the whole context, plus the image.
    assert len(teacher.requests) == 2
    assert len(teacher.requests[1]["messages"]) == 3
    assert teacher.requests[1]["images"] == [IMAGE_PHYSICAL]


def test_every_turn_consumes_exactly_one_budget_slot(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(tmp_path, [CODE_TURN, FINAL_TURN])

    result = builder.build_task(_task())

    assert result.budget_stats.consumed_slot == 2
    assert result.budget_stats.pending == 0

    ledger = TeacherRequestBudget(
        str(tmp_path / "teacher_requests.sqlite3"), "mulberry-stage1-000001"
    )
    rows = ledger.request_rows()
    assert [row["role"] for row in rows] == ["natural", "natural"]
    assert [row["status"] for row in rows] == ["successful", "successful"]


def test_teacher_timeout_is_recorded_without_a_retry(tmp_path) -> None:
    builder, teacher, _sandbox = _builder(tmp_path, [TimeoutError("too slow")])

    result = builder.build_task(_task())

    assert len(teacher.requests) == 1
    assert result.rollout.completed is False
    assert result.rollout.terminal_reason == "teacher_timeout"
    assert result.rollout.failed_request_status == "timeout"
    assert result.budget_stats.timeout == 1
    assert result.budget_stats.pending == 0


# --------------------------------------------------------------------------
# Validator gate and weak-label provenance
# --------------------------------------------------------------------------


def test_matching_reference_answer_exports_messages_and_images_only(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(tmp_path, [CODE_TURN, FINAL_TURN])

    result = builder.build_task(_task())

    assert result.rollout.answer_check.consistency == MATCH
    trajectory = result.trajectories[0]
    assert trajectory.final_answer_valid is True
    assert trajectory.clean_natural_lineage is True
    # No independent step validator exists in P0, so nothing is claimed here.
    assert trajectory.step_validated == ()

    decision, candidate = validate_trajectory_for_export(
        trajectory, source_guard=result.source_guard
    )
    assert decision.status == "accepted"
    assert decision.evidence_sources == ("deterministic_rule",)

    row = Projector().project(candidate)
    assert row is not None
    assert set(row.to_dict()) == {"messages", "images"}
    # The portable, image-root-relative path -- never the machine path.
    assert row.to_dict()["images"] == [IMAGE_RELATIVE]
    assert row.to_dict()["messages"] == [dict(item) for item in result.rollout.messages]


def test_audit_records_the_weak_label_provenance(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    result = builder.build_task(_task())

    assert {record.record_kind for record in result.audit_records} == {
        "task_analysis",
        "natural_rollout",
        "image_provenance",
        "reference_answer_check",
        "validation_decision",
    }
    check = _audit(result, "reference_answer_check")
    assert check["ground_truth_origin"] == GROUND_TRUTH_ORIGIN
    assert check["answer_check_method"] == "reference_answer_match_v1"
    assert check["independently_verified"] is False
    assert check["reference_label_is_weak"] is True
    assert _audit(result, "natural_rollout")["step_validation"] == STEP_VALIDATION_NOTE


def test_mismatching_reference_answer_is_not_exportable(tmp_path) -> None:
    wrong = "<think>wrong</think>\nCONFIDENCE: 0.5\nFINAL_ANSWER: 112"
    builder, _teacher, _sandbox = _builder(tmp_path, [wrong])

    result = builder.build_task(_task())

    assert result.rollout.answer_check.consistency == MISMATCH
    assert result.trajectories[0].final_answer_valid is False
    decision = _audit(result, "validation_decision")
    assert decision["status"] == "review_required"
    assert "final_answer_not_independently_validated" in decision["reasons"]
    assert result.candidates == ()


def test_final_answer_without_the_frozen_shape_is_rejected(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(
        tmp_path, ["<think>done</think>\nFINAL_ANSWER: 9"]
    )

    result = builder.build_task(_task())

    # The rollout terminates on the parser...
    assert result.rollout.terminal_reason == TERMINAL_FINAL_ANSWER
    assert result.rollout.answer_check.consistency == MATCH
    # ...but the frozen export gate still requires CONFIDENCE.
    decision = _audit(result, "validation_decision")
    assert decision["status"] == "review_required"
    assert any(
        reason.startswith("invalid_solver_final") for reason in decision["reasons"]
    )
    assert result.candidates == ()


# --------------------------------------------------------------------------
# Task Analysis scope
# --------------------------------------------------------------------------


def test_task_analysis_is_skipped_by_default(tmp_path) -> None:
    builder, teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    result = builder.build_task(_task())

    record = _audit(result, "task_analysis")
    assert record["status"] == "skipped"
    assert record["reason"] == "phase2b_p0_scope"
    assert result.trajectories[0].target_repair_depth_frozen == 0
    assert [request["role"] for request in teacher.requests] == ["natural"]


def test_task_analysis_freezes_the_target_depth_when_enabled(tmp_path) -> None:
    analysis = json.dumps({"target_repair_depth": 2})
    builder, teacher, _sandbox = _builder(
        tmp_path, [analysis, FINAL_TURN], enable_task_analysis=True
    )

    result = builder.build_task(_task())

    assert [request["role"] for request in teacher.requests] == [
        "task_analysis",
        "natural",
    ]
    trajectory = result.trajectories[0]
    assert trajectory.target_repair_depth == 2
    assert trajectory.target_repair_depth_frozen == 2


def test_task_analysis_parse_failure_stops_the_task(tmp_path) -> None:
    builder, teacher, _sandbox = _builder(
        tmp_path, ["not json at all", FINAL_TURN], enable_task_analysis=True
    )

    result = builder.build_task(_task())

    assert result.trajectories == ()
    assert result.candidates == ()
    assert result.budget_stats.parse_failed == 1
    assert [request["role"] for request in teacher.requests] == ["task_analysis"]


# --------------------------------------------------------------------------
# Clean lineage, source guard and manifest safety
# --------------------------------------------------------------------------


def test_root_snapshot_starts_from_the_scaffolded_question(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    result = builder.build_task(_task())

    root = result.root_snapshot
    assert root is not None
    assert root.parent_snapshot_id is None
    assert result.trajectories[0].root_snapshot_hash == root.snapshot_hash
    assert result.trajectories[0].start_snapshot_hash == root.snapshot_hash
    assert root.payload["original_images"] == [
        {"asset_id": "mulberry/item-1/image-0", "content_sha256": IMAGE_SHA}
    ]
    assert root.payload["messages"][0]["content"] == build_solver_user_content(
        _task().question
    )


def test_forbidden_partition_never_reaches_the_teacher(tmp_path) -> None:
    builder, teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    result = builder.build_task(_task(split="test", usage_partition="test"))

    assert teacher.requests == []
    assert result.root_snapshot is None
    assert result.candidates == ()
    assert result.audit_records[0].record_kind == "source_guard"


def test_failed_rollout_keeps_its_conversation_in_the_audit(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(tmp_path, ["<think>stuck</think>"])

    result = builder.build_task(_task())

    record = _audit(result, "natural_rollout")
    assert record["completed"] is False
    assert [item["role"] for item in record["messages"]] == ["user"]
    assert "Question: What is the value of the largest bar?" in record["messages"][0][
        "content"
    ]


def test_describe_is_manifest_safe(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    manifest = build_manifest(
        run_id="run-1",
        base_sha="a" * 40,
        extra=builder.describe(),
    )

    assert manifest["solver_prompt_version"] == SOLVER_PROMPT_VERSION
    assert manifest["ground_truth_origin"] == GROUND_TRUTH_ORIGIN
    assert manifest["independently_verified"] is False
    assert manifest["task_analysis_enabled"] is False


# --------------------------------------------------------------------------
# Portable image paths
# --------------------------------------------------------------------------


def test_export_images_prefer_the_relative_form() -> None:
    task = _task()

    assert task.images == (IMAGE_PHYSICAL,)
    assert task.export_images == (IMAGE_RELATIVE,)


def test_export_images_fall_back_to_images_for_relative_fixtures() -> None:
    task = _task(images=("fixture/item/image-0.png",), image_relatives=())

    assert task.export_images == ("fixture/item/image-0.png",)


def test_image_relatives_must_match_the_image_count() -> None:
    with pytest.raises(ValueError):
        _task(image_relatives=("a.png", "b.png"))


def test_image_provenance_audit_maps_physical_to_export_paths(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    result = builder.build_task(_task())

    record = _audit(result, "image_provenance")
    assert record["physical_paths"] == [IMAGE_PHYSICAL]
    assert record["export_paths"] == [IMAGE_RELATIVE]
    assert record["image_asset_ids"] == ["mulberry/item-1/image-0"]
    assert record["image_content_hashes"] == [IMAGE_SHA]
    assert record["image_placeholder_count"] == 1


def test_invalid_constructor_arguments_are_rejected(tmp_path) -> None:
    teacher = StubTeacher([FINAL_TURN])
    for kwargs in (
        {"max_reasoning_steps": 0},
        {"max_rollout_turns": 0},
        {"budget_limit": 0},
    ):
        with pytest.raises(ValueError):
            RealTrajectoryBuilder(
                backend=teacher,
                budget_db=str(tmp_path / "requests.sqlite3"),
                **kwargs,
            )
