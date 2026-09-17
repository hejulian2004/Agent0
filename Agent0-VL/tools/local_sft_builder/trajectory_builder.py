"""Real natural multi-turn Solver rollout for Phase 2B-P0.

This is the real counterpart of :class:`~.fake_builder.FakeTrajectoryBuilder`.
The state machine keeps the same shape -- accepted source, clean root
snapshot, optional Task Analysis, natural rollout, frozen Validator -- and only
the backend, runtime and source are swapped:

.. code-block:: text

    accepted SourceTask -> clean root snapshot -> natural rollout -> Validator

Three rules are load bearing:

1. **Termination is driven by the final-answer parser, not by a marker search
   or by the absence of a code block.**  A response is terminal only when
   :func:`~.answer_check.extract_final_answer` returns a value.  Otherwise the
   response must carry at least one executable fenced Python block; a response
   with neither is a hard failure with the reason
   ``solver_response_has_neither_final_answer_nor_executable_code``.  This is
   deliberately stricter than ``agent0_evaluator._generate_with_tools``, which
   also stops when no code block is present.

2. **The rollout never retries on its own.**  Every turn consumes one fresh
   ``TeacherRequestBudget`` slot; a retry is always a new slot, and sandbox
   execution costs zero slots.

3. **Weak labels are never presented as verification.**  ``final_answer_valid``
   is filled from the reference-answer consistency check only, and the audit
   trail records ``ground_truth_origin``, ``answer_check_method`` and
   ``independently_verified=false``.  ``step_validated`` stays empty because no
   independent step validator exists in Phase 2B-P0 (Verifier is deferred).

Protocol scaffolding
--------------------

The Mulberry and ReTool prompts ask for ``### The final answer is:`` and
``<answer>\\boxed{...}</answer>`` respectively; neither mentions ``<think>``,
``CONFIDENCE:`` or ``FINAL_ANSWER:``.  The frozen export gate
(:func:`~.protocol.validate_solver_final`) requires all three.  The rollout
therefore folds a protocol scaffold into the first *user* message, which is the
same technique the legacy builder used for its smoke rows.  The source question
is embedded verbatim so image placeholders keep their exact count and order,
and no ``system`` role is introduced because the frozen Validator only accepts
``user``/``assistant``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .answer_check import (
    AnswerCheckResult,
    check_reference_answer,
    describe as describe_answer_check,
    extract_final_answer,
)
from .budget import BudgetStats, RequestResult, TeacherRequestBudget
from .fake_builder import SourceTask
from .protocol import (
    ProtocolError,
    extract_python_blocks,
    reject_json_tool_call,
)
from .runtime import SourceRuntimeAdapter
from .schema import AuditRecord, ExportCandidate, TrajectoryRecord
from .snapshots import ImmutableSnapshot
from .source_guard import SourceGuardDecision, SourceLeakageGuard
# Single source of truth: the provenance label is stamped onto source rows by
# the normalizer and must never drift from what the audit trail claims.
from .source_normalize import GROUND_TRUTH_ORIGIN
from .validator import validate_trajectory_for_export


BUILDER_VERSION = "agent0vl.local_sft_builder.trajectory_builder.v1"
SOLVER_PROMPT_VERSION = "agent0vl.local_sft_builder.solver_prompt.v1"

DEFAULT_MAX_REASONING_STEPS = 8
DEFAULT_BUDGET_LIMIT = 32

TERMINAL_FINAL_ANSWER = "final_answer_parsed"
FAILURE_NO_TERMINAL = (
    "solver_response_has_neither_final_answer_nor_executable_code"
)
FAILURE_MAX_TURNS = "solver_exceeded_max_rollout_turns"
FAILURE_JSON_TOOL_CALL = "solver_json_tool_call_is_not_source_compatible"
FAILURE_TEACHER_PREFIX = "teacher_"

# Phase 2B-P0 performs no independent step validation: the Verifier role is
# deferred, so ``TrajectoryRecord.step_validated`` stays empty rather than
# claiming that the sandbox validated each step.
STEP_VALIDATION_NOTE = "not_performed_phase2b_p0"

SOLVER_PROMPT_SCAFFOLD = (
    "## Task protocol\n"
    "Solve the problem step by step.\n"
    "- Wrap each reasoning step in <think>...</think> tags.\n"
    "- If you need a computation, write Python in a fenced block:\n"
    "  ```python\n"
    "  # your code\n"
    "  ```\n"
    "  The code runs in an external sandbox and its output is returned to you "
    "as a [Code Execution Result] message. Do not emit JSON tool calls.\n"
    "- When you are finished, end your response with exactly these two lines:\n"
    "  CONFIDENCE: <a number between 0 and 1>\n"
    "  FINAL_ANSWER: <your final answer on one line>\n"
)

ANALYSIS_PROMPT = (
    "You are a task analyst for a tool-using vision-language agent.\n"
    "Return exactly one JSON object, on one line, with the single key\n"
    '"target_repair_depth" whose value is 0, 1 or 2.\n'
)


def build_solver_user_content(question: str) -> str:
    """Fold the protocol scaffold into the first user message.

    The source question is embedded verbatim, so the number and order of
    ``<image>`` placeholders is preserved exactly.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return f"{SOLVER_PROMPT_SCAFFOLD}\n{question}"


def _response_text(value: Any) -> str:
    """Unwrap a Teacher response into text, tolerating scripted backends."""

    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(value, str):
        return value
    raise TypeError("teacher response must be text or expose a text attribute")


@dataclass(frozen=True)
class RolloutTurn:
    """One Teacher turn inside a natural rollout."""

    turn_index: int
    request_id: str
    status: str
    final_answer: str | None = None
    code_block_count: int = 0
    executed_block_count: int = 0
    sandbox_statuses: tuple[str, ...] = ()
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "request_id": self.request_id,
            "status": self.status,
            "final_answer": self.final_answer,
            "code_block_count": self.code_block_count,
            "executed_block_count": self.executed_block_count,
            "sandbox_statuses": list(self.sandbox_statuses),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class RolloutOutcome:
    """Result of one natural rollout, complete or failed."""

    task_id: str
    completed: bool
    terminal_reason: str
    turns: tuple[RolloutTurn, ...]
    messages: tuple[dict[str, Any], ...]
    executed_step_count: int = 0
    final_assistant_text: str | None = None
    answer_check: AnswerCheckResult | None = None
    failed_request_status: str | None = None
    error: str | None = None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "completed": self.completed,
            "terminal_reason": self.terminal_reason,
            "turn_count": self.turn_count,
            "executed_step_count": self.executed_step_count,
            "message_count": len(self.messages),
            "failed_request_status": self.failed_request_status,
            "error": self.error,
            "step_validation": STEP_VALIDATION_NOTE,
            "turns": [turn.to_dict() for turn in self.turns],
        }


@dataclass(frozen=True)
class RealBuildResult:
    """Everything one real task produced, including its audit trail."""

    task: SourceTask
    root_snapshot: ImmutableSnapshot | None
    trajectories: tuple[TrajectoryRecord, ...]
    audit_records: tuple[AuditRecord, ...]
    candidates: tuple[ExportCandidate, ...]
    source_guard: SourceGuardDecision
    budget_stats: BudgetStats
    rollout: RolloutOutcome | None = None

    @property
    def accepted(self) -> bool:
        return bool(self.candidates)


class RealTrajectoryBuilder:
    """Run one natural, source-compatible Solver rollout per task."""

    def __init__(
        self,
        *,
        backend: Any,
        budget_db: str,
        sandbox_runtime: SourceRuntimeAdapter | None = None,
        source_guard: SourceLeakageGuard | None = None,
        base_sha: str | None = None,
        max_reasoning_steps: int = DEFAULT_MAX_REASONING_STEPS,
        max_rollout_turns: int | None = None,
        budget_limit: int = DEFAULT_BUDGET_LIMIT,
        enable_task_analysis: bool = False,
        sampling: Any | None = None,
    ):
        if max_reasoning_steps <= 0:
            raise ValueError("max_reasoning_steps must be positive")
        if max_rollout_turns is not None and max_rollout_turns <= 0:
            raise ValueError("max_rollout_turns must be positive")
        if budget_limit <= 0:
            raise ValueError("budget_limit must be positive")
        self.backend = backend
        self.runtime = sandbox_runtime or SourceRuntimeAdapter()
        self.budget_db = budget_db
        self.source_guard = source_guard or SourceLeakageGuard()
        # ``ImmutableSnapshot`` requires a 64-character digest here, so the
        # 40-character git commit SHA cannot be passed directly; the manifest
        # records the commit SHA separately.  Supply a digest or None.
        self.base_sha = base_sha
        self.max_reasoning_steps = max_reasoning_steps
        self.max_rollout_turns = max_rollout_turns or max_reasoning_steps
        self.budget_limit = budget_limit
        self.enable_task_analysis = enable_task_analysis
        self.sampling = sampling

    def describe(self) -> dict[str, Any]:
        """Non-secret provenance for the run manifest."""

        return {
            "builder_version": BUILDER_VERSION,
            "solver_prompt_version": SOLVER_PROMPT_VERSION,
            "solver_prompt_has_system_role": False,
            "max_rollout_turns": self.max_rollout_turns,
            "max_reasoning_steps": self.max_reasoning_steps,
            "budget_limit_per_task": self.budget_limit,
            "task_analysis_enabled": self.enable_task_analysis,
            "step_validation": STEP_VALIDATION_NOTE,
            "ground_truth_origin": GROUND_TRUTH_ORIGIN,
            **describe_answer_check(),
        }

    # -- request plumbing ---------------------------------------------------

    def _request(
        self,
        budget: TeacherRequestBudget,
        task: SourceTask,
        role: str,
        messages: list[dict[str, Any]],
    ) -> RequestResult:
        """Consume one budget slot for exactly one Teacher attempt."""

        return budget.execute(
            role=role,
            backend=self.backend.generate_from_payload,
            payload={
                "role": role,
                "messages": list(messages),
                "images": list(task.images),
                "sampling": self.sampling,
            },
        )

    def _root_snapshot(self, task: SourceTask) -> ImmutableSnapshot:
        return ImmutableSnapshot.create(
            messages=[
                {
                    "role": "user",
                    "content": build_solver_user_content(task.question),
                }
            ],
            original_images=list(task.image_refs),
            generation_boundaries=[{"boundary": "root", "step": 0}],
            sandbox_state_mode="stateless",
            base_sha=self.base_sha,
        )

    # -- optional Task Analysis --------------------------------------------

    @staticmethod
    def _analysis_parser(value: Any) -> dict[str, Any]:
        parsed = json.loads(_response_text(value))
        if not isinstance(parsed, dict):
            raise ValueError("Task Analysis must return an object")
        depth = parsed.get("target_repair_depth")
        if isinstance(depth, bool) or not isinstance(depth, int) or not 0 <= depth <= 2:
            raise ValueError("target_repair_depth must be 0, 1 or 2")
        return parsed

    def _analysis(
        self,
        budget: TeacherRequestBudget,
        task: SourceTask,
    ) -> tuple[int, dict[str, Any]] | None:
        messages = [{"role": "user", "content": f"{ANALYSIS_PROMPT}\n{task.question}"}]
        result = budget.execute(
            role="task_analysis",
            backend=self.backend.generate_from_payload,
            payload={
                "role": "task_analysis",
                "messages": messages,
                "images": list(task.images),
                "sampling": self.sampling,
            },
            parser=self._analysis_parser,
        )
        if result.status != "successful":
            return None
        analysis = result.value
        return int(analysis["target_repair_depth"]), analysis

    # -- natural rollout ----------------------------------------------------

    def _rollout(
        self,
        budget: TeacherRequestBudget,
        task: SourceTask,
    ) -> RolloutOutcome:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_solver_user_content(task.question)}
        ]
        turns: list[RolloutTurn] = []
        executed_steps = 0

        for turn_index in range(self.max_rollout_turns):
            result = self._request(budget, task, "natural", messages)
            if result.status != "successful":
                return RolloutOutcome(
                    task_id=task.task_id,
                    completed=False,
                    terminal_reason=f"{FAILURE_TEACHER_PREFIX}{result.status}",
                    turns=tuple(turns),
                    messages=tuple(messages),
                    executed_step_count=executed_steps,
                    failed_request_status=result.status,
                    error=result.error,
                )
            text = _response_text(result.value)

            # Fail closed on the JSON tool-call shape: the frozen runtime only
            # executes fenced Python blocks.
            try:
                reject_json_tool_call(text)
            except ProtocolError as exc:
                return RolloutOutcome(
                    task_id=task.task_id,
                    completed=False,
                    terminal_reason=FAILURE_JSON_TOOL_CALL,
                    turns=tuple(turns),
                    messages=tuple(messages),
                    executed_step_count=executed_steps,
                    error=str(exc),
                )

            # 1) The final-answer parser is the only termination authority.
            final_answer = extract_final_answer(text)
            if final_answer is not None:
                messages.append({"role": "assistant", "content": text})
                turns.append(
                    RolloutTurn(
                        turn_index=turn_index,
                        request_id=result.request_id,
                        status=result.status,
                        final_answer=final_answer,
                        terminal=True,
                    )
                )
                return RolloutOutcome(
                    task_id=task.task_id,
                    completed=True,
                    terminal_reason=TERMINAL_FINAL_ANSWER,
                    turns=tuple(turns),
                    messages=tuple(messages),
                    executed_step_count=executed_steps,
                    final_assistant_text=text,
                )

            # 2) No final answer means the turn must carry executable code.
            blocks = extract_python_blocks(text)
            if not blocks:
                return RolloutOutcome(
                    task_id=task.task_id,
                    completed=False,
                    terminal_reason=FAILURE_NO_TERMINAL,
                    turns=tuple(turns),
                    messages=tuple(messages),
                    executed_step_count=executed_steps,
                )

            calls, results, observation = self.runtime.execute_solver_text(text)
            if observation is None:  # pragma: no cover - blocks imply execution
                raise AssertionError("executed blocks must produce an observation")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": observation})
            executed_steps += len(calls)
            turns.append(
                RolloutTurn(
                    turn_index=turn_index,
                    request_id=result.request_id,
                    status=result.status,
                    code_block_count=len(blocks),
                    executed_block_count=len(calls),
                    sandbox_statuses=tuple(
                        str(item.get("status")) for item in results
                    ),
                )
            )

        return RolloutOutcome(
            task_id=task.task_id,
            completed=False,
            terminal_reason=FAILURE_MAX_TURNS,
            turns=tuple(turns),
            messages=tuple(messages),
            executed_step_count=executed_steps,
        )

    # -- task assembly ------------------------------------------------------

    @staticmethod
    def _audit(
        task: SourceTask,
        record_id: str,
        kind: str,
        payload: Mapping[str, Any],
    ) -> AuditRecord:
        return AuditRecord(
            record_id=record_id,
            task_id=task.task_id,
            record_kind=kind,
            payload=dict(payload),
            supervision_status="audit_only",
        )

    def _empty_result(
        self,
        task: SourceTask,
        root: ImmutableSnapshot | None,
        audit_records: list[AuditRecord],
        source_decision: SourceGuardDecision,
        budget: TeacherRequestBudget,
        rollout: RolloutOutcome | None,
    ) -> RealBuildResult:
        return RealBuildResult(
            task=task,
            root_snapshot=root,
            trajectories=(),
            audit_records=tuple(audit_records),
            candidates=(),
            source_guard=source_decision,
            budget_stats=budget.assert_consistent(require_no_pending=True),
            rollout=rollout,
        )

    def build_task(self, task: SourceTask) -> RealBuildResult:
        source_decision = self.source_guard.check(
            task.as_source_record(), expected_stage=task.stage
        )
        budget = TeacherRequestBudget(
            self.budget_db, task.task_id, limit=self.budget_limit
        )
        audit_records: list[AuditRecord] = []

        # Crash recovery: a previous run may have left a pending slot behind.
        recovered = budget.recover_stale_pending()
        if recovered:
            audit_records.append(
                self._audit(
                    task,
                    f"{task.task_id}:budget_recovery",
                    "budget_recovery",
                    {"recovered_pending": recovered},
                )
            )

        if not source_decision.accepted:
            audit_records.append(
                self._audit(
                    task,
                    f"{task.task_id}:source_guard",
                    "source_guard",
                    {
                        "status": source_decision.status,
                        "reasons": list(source_decision.reasons),
                    },
                )
            )
            return self._empty_result(
                task, None, audit_records, source_decision, budget, None
            )

        root = self._root_snapshot(task)

        target_depth = 0
        if self.enable_task_analysis:
            analysis_result = self._analysis(budget, task)
            if analysis_result is None:
                audit_records.append(
                    self._audit(
                        task,
                        f"{task.task_id}:task_analysis",
                        "task_analysis",
                        {"status": "review_required"},
                    )
                )
                return self._empty_result(
                    task, root, audit_records, source_decision, budget, None
                )
            target_depth, analysis = analysis_result
            audit_records.append(
                self._audit(
                    task,
                    f"{task.task_id}:task_analysis",
                    "task_analysis",
                    {"status": "successful", "analysis": analysis},
                )
            )
        else:
            audit_records.append(
                self._audit(
                    task,
                    f"{task.task_id}:task_analysis",
                    "task_analysis",
                    {
                        "status": "skipped",
                        "reason": "phase2b_p0_scope",
                        "target_repair_depth": target_depth,
                    },
                )
            )

        outcome = self._rollout(budget, task)
        rollout_payload = outcome.to_dict()
        if not outcome.completed:
            # A failed rollout has no trajectory row, so the audit record is the
            # only place its conversation survives.
            rollout_payload["messages"] = [
                dict(message) for message in outcome.messages
            ]
        audit_records.append(
            self._audit(
                task,
                f"{task.task_id}:natural_rollout",
                "natural_rollout",
                rollout_payload,
            )
        )
        audit_records.append(
            self._audit(
                task,
                f"{task.task_id}:image_provenance",
                "image_provenance",
                {
                    "physical_paths": list(task.images),
                    "export_paths": list(task.export_images),
                    "image_asset_ids": [
                        ref["asset_id"] for ref in task.image_refs
                    ],
                    "image_content_hashes": list(task.image_content_hashes),
                    "image_placeholder_count": sum(
                        message["content"].count("<image>")
                        for message in outcome.messages
                    ),
                },
            )
        )

        if not outcome.completed:
            return self._empty_result(
                task, root, audit_records, source_decision, budget, outcome
            )

        check = check_reference_answer(
            outcome.final_assistant_text or "", task.ground_truth
        )
        outcome = RolloutOutcome(
            task_id=outcome.task_id,
            completed=outcome.completed,
            terminal_reason=outcome.terminal_reason,
            turns=outcome.turns,
            messages=outcome.messages,
            executed_step_count=outcome.executed_step_count,
            final_assistant_text=outcome.final_assistant_text,
            answer_check=check,
            failed_request_status=outcome.failed_request_status,
            error=outcome.error,
        )
        audit_records.append(
            self._audit(
                task,
                f"{task.task_id}:reference_answer_check",
                "reference_answer_check",
                {
                    "ground_truth_origin": GROUND_TRUTH_ORIGIN,
                    "independently_verified": False,
                    "reference_label_is_weak": True,
                    **check.to_dict(),
                },
            )
        )

        trajectory = TrajectoryRecord(
            task_id=task.task_id,
            trajectory_id=f"{task.task_id}:natural-1",
            source_record_id=task.source_record_id,
            stage=task.stage,
            messages=outcome.messages,
            images=task.export_images,
            image_content_hashes=task.image_content_hashes,
            canonical_source="natural",
            lineage_events=("natural",),
            ancestor_trajectory_ids=(),
            root_snapshot_hash=root.snapshot_hash,
            start_snapshot_hash=root.snapshot_hash,
            controlled_intervention=False,
            audit_context_received=False,
            final_answer_valid=check.matched,
            step_validated=(),
            target_repair_depth=target_depth,
            target_repair_depth_frozen=target_depth,
            canonical_actual_repair_count=0,
            branch_actual_repair_count=0,
        )
        decision, candidate = validate_trajectory_for_export(
            trajectory,
            source_guard=source_decision,
            max_reasoning_steps=self.max_reasoning_steps,
        )
        audit_records.append(
            self._audit(
                task,
                f"{task.task_id}:validation",
                "validation_decision",
                decision.to_dict(),
            )
        )

        return RealBuildResult(
            task=task,
            root_snapshot=root,
            trajectories=(trajectory,),
            audit_records=tuple(audit_records),
            candidates=(candidate,) if candidate is not None else (),
            source_guard=source_decision,
            budget_stats=budget.assert_consistent(require_no_pending=True),
            rollout=outcome,
        )
