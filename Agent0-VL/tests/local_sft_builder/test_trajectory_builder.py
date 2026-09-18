"""Tests for the real natural rollout builder (Phase 2B-P0).

Everything here is synthetic: a scripted Teacher, a recording sandbox and an
in-memory task.  No dataset, endpoint or real sandbox is touched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools.local_sft_builder.answer_check import ANSWER_CHECK_METHOD, MATCH, MISMATCH
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
    SOLVER_PROMPT_ADDENDUM,
    SOLVER_PROMPT_ADDENDUM_VERSION,
    SOLVER_PROMPT_VERSION,
    SOLVER_SYSTEM_PROMPT,
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
# Solver protocol: system prompt, not a local user-turn scaffold
# --------------------------------------------------------------------------


def _upstream_evaluator_system_prompt() -> str:
    """Extract ``_build_prompt``'s default system prompt from the upstream file.

    Parsing the assignment with ``ast`` resolves Python's implicit string
    concatenation, so this compares the real runtime string rather than a
    whitespace-normalized approximation.  Importing the module itself is not an
    option: it pulls in the full vLLM/verl stack.
    """

    evaluator = (
        Path(__file__).resolve().parents[2]
        / "verl"
        / "evaluation"
        / "agent0_evaluator.py"
    )
    tree = ast.parse(evaluator.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "system_prompt":
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
    raise AssertionError("system_prompt assignment not found in agent0_evaluator.py")


def test_solver_system_prompt_matches_the_upstream_runtime() -> None:
    """The protocol is the source's, so it must not drift from it."""

    upstream = _upstream_evaluator_system_prompt()

    assert SOLVER_SYSTEM_PROMPT == upstream
    assert "```python" in SOLVER_SYSTEM_PROMPT
    assert "\\boxed{...}" in SOLVER_SYSTEM_PROMPT
    # The local triple is not part of the source Solver protocol.
    for marker in ("<think>", "CONFIDENCE:", "FINAL_ANSWER:"):
        assert marker not in SOLVER_SYSTEM_PROMPT


def test_user_content_is_the_source_question_verbatim() -> None:
    question = "<image>Question: What is the value of the largest bar?"

    content = build_solver_user_content(question)

    assert content == question
    assert content.count("<image>") == question.count("<image>") == 1
    for marker in ("<think>", "```python", "CONFIDENCE:", "FINAL_ANSWER:"):
        assert marker not in content


def test_scaffold_rejects_an_empty_question() -> None:
    with pytest.raises(ValueError):
        build_solver_user_content("   ")


def test_scaffold_preserves_multiple_image_placeholders() -> None:
    question = "<image>Compare <image> and answer."

    content = build_solver_user_content(question)

    assert content.count("<image>") == 2


def test_solver_system_prompt_is_sent_but_never_exported(tmp_path) -> None:
    """The system message belongs to the request, not to the training row."""

    builder, teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    result = builder.build_task(_task())

    request_messages = teacher.requests[0]["messages"]
    assert request_messages[0] == {
        "role": "system",
        "content": SOLVER_SYSTEM_PROMPT,
    }
    assert [message["role"] for message in request_messages[1:]] == ["user"]

    exported_roles = {
        message["role"] for message in result.rollout.messages
    }
    assert exported_roles == {"user", "assistant"}
    assert result.candidates
    assert "system" not in {
        message["role"] for message in result.candidates[0].messages
    }


def test_solver_system_prompt_can_be_disabled(tmp_path) -> None:
    builder, teacher, _sandbox = _builder(
        tmp_path, [FINAL_TURN], solver_system_prompt=None
    )

    builder.build_task(_task())

    assert [message["role"] for message in teacher.requests[0]["messages"]] == [
        "user"
    ]


def test_prompt_addendum_rides_the_system_message_only(tmp_path) -> None:
    """The local suffix must never reach the exported row.

    Training re-supplies the upstream instruction through ``--system``; a local
    suffix inside the row would silently change what the student is trained on.
    """

    addendum = "\n\nEnvironment notes for this task:\n- local clause"
    builder, teacher, _sandbox = _builder(
        tmp_path, [FINAL_TURN], solver_prompt_addendum=addendum
    )

    result = builder.build_task(_task())

    assert teacher.requests[0]["messages"][0] == {
        "role": "system",
        "content": SOLVER_SYSTEM_PROMPT + addendum,
    }
    exported = "\n".join(message["content"] for message in result.rollout.messages)
    assert "local clause" not in exported


def test_builder_sends_no_addendum_by_default(tmp_path) -> None:
    """Unit tests and pre-addendum comparison runs keep the upstream prompt."""

    builder, _teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    assert builder.solver_prompt_addendum is None
    assert builder.effective_solver_system_prompt == SOLVER_SYSTEM_PROMPT


def test_manifest_records_the_addendum_separately_from_the_protocol(tmp_path) -> None:
    addendum = "\n\nEnvironment notes for this task:\n- local clause"
    builder, _teacher, _sandbox = _builder(
        tmp_path, [FINAL_TURN], solver_prompt_addendum=addendum
    )

    described = builder.describe()

    assert described["solver_system_prompt_exported"] is False
    assert (
        described["solver_prompt_addendum_version"]
        == SOLVER_PROMPT_ADDENDUM_VERSION
    )
    assert (
        described["solver_prompt_addendum_sha256"]
        != described["solver_system_prompt_sha256"]
    )
    assert described["effective_solver_system_prompt_sha256"] is not None


def test_manifest_marks_the_addendum_absent_when_disabled(tmp_path) -> None:
    builder, _teacher, _sandbox = _builder(tmp_path, [FINAL_TURN])

    described = builder.describe()

    assert described["solver_prompt_addendum_sha256"] is None
    assert described["solver_prompt_addendum_version"] is None
    assert (
        described["effective_solver_system_prompt_sha256"]
        == described["solver_system_prompt_sha256"]
    )


def test_default_addendum_states_the_sandbox_cannot_see_the_image() -> None:
    """The clause exists because the upstream prompt invites the impossible.

    ``SOLVER_SYSTEM_PROMPT`` tells the Teacher it may "manipulate the image
    (crop, resize, adjust contrast)", but the sandbox has no image file and no
    matplotlib/cv2.  Measured, that mismatch produced compliance theatre such as
    ``Image.open("chart.png") if False else None``.
    """

    assert SOLVER_SYSTEM_PROMPT.count("manipulate the image") == 1
    assert "no image file" in SOLVER_PROMPT_ADDENDUM
    assert "matplotlib" in SOLVER_PROMPT_ADDENDUM
    # The turn-discipline clause is what makes the code reach the sandbox at all.
    assert "must NOT contain the final answer" in SOLVER_PROMPT_ADDENDUM


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


def test_a_turn_with_code_and_an_answer_is_salvaged_into_a_tool_call(
    tmp_path,
) -> None:
    """The premature answer is dropped and the code is run instead.

    Terminating on the answer -- the upstream behaviour -- exports the code block
    unexecuted, because the exported row is ``record.messages`` verbatim.
    Measured on a 200-task run, 18.6% of exported rows carried a code block that
    never ran, which teaches "write code, then answer without waiting".

    Salvaging runs the code and re-asks, so the row ends with an answer that was
    genuinely produced with the observation in context.
    """

    builder, teacher, sandbox = _builder(
        tmp_path, [f"{FINAL_TURN}\n{CODE_TURN}", FINAL_TURN]
    )

    result = builder.build_task(_task())

    assert sandbox.calls == ["print(9)"]
    assert len(teacher.requests) == 2
    assert result.rollout.completed is True
    assert result.rollout.terminal_reason == TERMINAL_FINAL_ANSWER

    roles = [message["role"] for message in result.rollout.messages]
    assert roles == ["user", "assistant", "user", "assistant"]

    salvaged_turn = result.rollout.messages[1]["content"]
    # The premature answer is gone...
    assert "FINAL_ANSWER" not in salvaged_turn
    # ...while the reasoning and the code that justified the salvage survive.
    assert "<think>The tallest bar reaches 9.</think>" in salvaged_turn
    assert "```python\nprint(9)\n```" in salvaged_turn
    assert "[Code Execution Result]" in result.rollout.messages[2]["content"]

    # The re-request carried the sandbox observation, which is the whole point.
    assert any(
        "[Code Execution Result]" in str(message.get("content"))
        for message in teacher.requests[1]["messages"]
    )
    assert result.rollout.turns[0].salvaged is True
    assert result.rollout.executed_step_count == 1
    # A salvaged trajectory is still a valid export: the final turn is the fresh
    # answer, and the intervening observation is an ordinary ``user`` turn.
    assert result.candidates
    assert [m["role"] for m in result.candidates[0].messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_a_second_violation_falls_back_to_the_upstream_termination(
    tmp_path,
) -> None:
    """Salvage is capped at one per task, so the extra request cost is bounded."""

    builder, _teacher, sandbox = _builder(
        tmp_path,
        [f"{FINAL_TURN}\n{CODE_TURN}", f"{FINAL_TURN}\n{CODE_TURN}"],
    )

    result = builder.build_task(_task())

    assert sandbox.calls == ["print(9)"]
    assert result.rollout.terminal_reason == TERMINAL_FINAL_ANSWER
    assert [message["role"] for message in result.rollout.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert result.rollout.turns[0].salvaged is True
    assert result.rollout.turns[-1].salvaged is False
    assert result.rollout.turns[-1].terminal is True


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
    # The system prompt is prepended to every request but is never part of the
    # exported rollout, so 4 = system + user + assistant + user.
    assert len(teacher.requests) == 2
    assert len(teacher.requests[1]["messages"]) == 4
    assert teacher.requests[1]["messages"][0]["role"] == "system"
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
    assert check["answer_check_method"] == ANSWER_CHECK_METHOD
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


def test_final_answer_without_the_local_confidence_marker_is_exportable(
    tmp_path,
) -> None:
    """``CONFIDENCE:`` is local metadata, not part of the upstream protocol.

    The upstream Solver prompt (``agent0_evaluator._build_prompt``) asks only for
    a final answer; nothing in the author runtime or training entry requires a
    confidence line.  The export gate must therefore not invent one.
    """

    builder, _teacher, _sandbox = _builder(
        tmp_path, ["<think>done</think>\nFINAL_ANSWER: 9"]
    )

    result = builder.build_task(_task())

    assert result.rollout.terminal_reason == TERMINAL_FINAL_ANSWER
    assert result.rollout.answer_check.consistency == MATCH
    decision = _audit(result, "validation_decision")
    assert decision["status"] == "accepted"
    assert decision["exportable"] is True
    assert len(result.candidates) == 1


def test_boxed_final_answer_in_the_author_format_is_exportable(tmp_path) -> None:
    """End-to-end proof that the author's ``\\boxed{...}`` shape reaches export."""

    builder, _teacher, _sandbox = _builder(
        tmp_path, ["<think>The bar reaches 9.</think>\n\\boxed{9}"]
    )

    result = builder.build_task(_task())

    assert result.rollout.terminal_reason == TERMINAL_FINAL_ANSWER
    assert result.rollout.answer_check.consistency == MATCH
    decision = _audit(result, "validation_decision")
    assert decision["status"] == "accepted"
    assert decision["exportable"] is True
    assert len(result.candidates) == 1


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
