"""Scripted three-task builder used to prove the local construction contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .budget import RequestResult, TeacherRequestBudget
from .protocol import parse_repair_json, parse_verifier_json
from .runtime import ScriptedBackend, SourceRuntimeAdapter
from .schema import AuditRecord, SupervisionUnit, TrajectoryRecord
from .snapshots import ImmutableSnapshot
from .source_guard import SourceGuardDecision, SourceIdentity, SourceLeakageGuard


@dataclass(frozen=True)
class SourceTask:
    task_id: str
    source_record_id: str
    source_dataset: str
    stage: str
    question: str
    images: tuple[str, ...] = ()
    image_refs: tuple[dict[str, str], ...] = ()
    ground_truth: str = ""
    split: str = "train"
    usage_partition: str | None = None
    source_revision: str = "fixture-v1"

    def __post_init__(self) -> None:
        if len(self.images) != len(self.image_refs):
            raise ValueError(
                "images and image_refs must contain the same number of items"
            )
        for index, image_ref in enumerate(self.image_refs):
            if "content_sha256" not in image_ref:
                raise ValueError(
                    f"image_refs[{index}] requires content_sha256"
                )

    @property
    def image_content_hashes(self) -> tuple[str, ...]:
        """Return ordered image hashes from the canonical image references."""

        return tuple(ref["content_sha256"] for ref in self.image_refs)

    def as_source_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_record_id": self.source_record_id,
            "original_id": self.source_record_id,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "question": self.question,
            "image_count": len(self.images),
            "image_content_hashes": list(self.image_content_hashes),
            "split": self.split,
            "usage_partition": self.usage_partition or self.stage,
        }


@dataclass(frozen=True)
class FakeBuildResult:
    task: SourceTask
    root_snapshot: ImmutableSnapshot | None
    trajectories: tuple[TrajectoryRecord, ...]
    audit_records: tuple[AuditRecord, ...]
    candidates: tuple[Any, ...]
    source_guard: SourceGuardDecision
    budget_stats: Any


class FakeTrajectoryBuilder:
    """Build deterministic A/B/C scenarios without a real Teacher model."""

    def __init__(
        self,
        *,
        backend: ScriptedBackend,
        sandbox_runtime: SourceRuntimeAdapter | None = None,
        budget_db: str,
        source_guard: SourceLeakageGuard | None = None,
        base_sha: str | None = None,
        max_reasoning_steps: int = 8,
    ):
        self.backend = backend
        self.runtime = sandbox_runtime or SourceRuntimeAdapter()
        self.budget_db = budget_db
        self.source_guard = source_guard or SourceLeakageGuard()
        self.base_sha = base_sha
        self.max_reasoning_steps = max_reasoning_steps

    def _request(
        self,
        budget: TeacherRequestBudget,
        task: SourceTask,
        phase: str,
        *,
        parser: Any = None,
    ) -> RequestResult:
        return budget.execute(
            role=phase,
            backend=self.backend.generate,
            payload={"task_id": task.task_id, "phase": f"{task.task_id}:{phase}"},
            parser=parser,
        )

    def _root_snapshot(self, task: SourceTask) -> ImmutableSnapshot:
        return ImmutableSnapshot.create(
            messages=[{"role": "user", "content": task.question}],
            original_images=list(task.image_refs),
            generation_boundaries=[{"boundary": "root", "step": 0}],
            sandbox_state_mode="stateless",
            base_sha=self.base_sha,
        )

    @staticmethod
    def _analysis_parser(value: str) -> dict[str, Any]:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Task Analysis must return an object")
        depth = parsed.get("target_repair_depth")
        if not isinstance(depth, int) or not 0 <= depth <= 2:
            raise ValueError("target_repair_depth must be 0, 1 or 2")
        return parsed

    def _analysis(
        self,
        budget: TeacherRequestBudget,
        task: SourceTask,
    ) -> tuple[int, dict[str, Any]] | None:
        result = self._request(budget, task, "task_analysis", parser=self._analysis_parser)
        if result.status != "successful":
            return None
        analysis = result.value
        return int(analysis["target_repair_depth"]), analysis

    def _solver_messages(
        self,
        task: SourceTask,
        solver_text: str,
    ) -> tuple[dict[str, str], ...]:
        messages: list[dict[str, str]] = [
            {"role": "user", "content": task.question}
        ]
        calls, results, observation = self.runtime.execute_solver_text(solver_text)
        messages.append({"role": "assistant", "content": solver_text})
        if observation is not None:
            messages.append({"role": "user", "content": observation})
        return tuple(messages)

    @staticmethod
    def _unit(
        *,
        task: SourceTask,
        trajectory_id: str,
        suffix: str,
        supervision_type: str,
        context: tuple[dict[str, Any], ...],
        target: str,
        evidence: tuple[str, ...],
        controlled: bool = False,
        repaired_error: bool = False,
        regenerated: bool = False,
    ) -> SupervisionUnit:
        return SupervisionUnit(
            record_id=f"{trajectory_id}:{suffix}",
            task_id=task.task_id,
            trajectory_id=trajectory_id,
            supervision_type=supervision_type,
            context_messages=context,
            assistant_target=target,
            evidence_sources=evidence,
            controlled_ancestry=controlled,
            repaired_error=repaired_error,
            regenerated=regenerated,
        )

    def _audit_record(
        self,
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

    def build_task(self, task: SourceTask) -> FakeBuildResult:
        source_decision = self.source_guard.check(
            task.as_source_record(), expected_stage=task.stage
        )
        budget = TeacherRequestBudget(self.budget_db, task.task_id, limit=32)
        if not source_decision.accepted:
            audit = self._audit_record(
                task,
                f"{task.task_id}:source_guard",
                "source_guard",
                {"status": source_decision.status, "reasons": source_decision.reasons},
            )
            return FakeBuildResult(
                task,
                None,
                (),
                (audit,),
                (),
                source_decision,
                budget.assert_consistent(require_no_pending=True),
            )
        root = self._root_snapshot(task)

        analysis_result = self._analysis(budget, task)
        if analysis_result is None:
            audit = self._audit_record(
                task,
                f"{task.task_id}:analysis",
                "task_analysis",
                {"status": "review_required"},
            )
            return FakeBuildResult(
                task,
                root,
                (),
                (audit,),
                (),
                source_decision,
                budget.assert_consistent(require_no_pending=True),
            )
        target_depth, analysis = analysis_result

        natural_result = self._request(budget, task, "natural")
        if natural_result.status != "successful":
            audit = self._audit_record(
                task,
                f"{task.task_id}:natural_failure",
                "natural_rollout",
                {"status": natural_result.status, "error": natural_result.error},
            )
            return FakeBuildResult(
                task,
                root,
                (),
                (audit,),
                (),
                source_decision,
                budget.assert_consistent(require_no_pending=True),
            )
        natural_text = str(natural_result.value)
        natural_messages = self._solver_messages(task, natural_text)
        trajectory_id = f"{task.task_id}:natural-1"
        trajectories: list[TrajectoryRecord] = []
        audit_records: list[AuditRecord] = []
        candidates: list[Any] = []

        # Direct success: no audit context is introduced into the lineage.
        if task.task_id.endswith("A"):
            trajectory = TrajectoryRecord(
                task_id=task.task_id,
                trajectory_id=trajectory_id,
                source_record_id=task.source_record_id,
                stage=task.stage,
                messages=natural_messages,
                images=task.images,
                image_content_hashes=task.image_content_hashes,
                canonical_source="natural",
                lineage_events=("natural",),
                ancestor_trajectory_ids=(),
                root_snapshot_hash=root.snapshot_hash,
                start_snapshot_hash=root.snapshot_hash,
                final_answer_valid=True,
                step_validated=(True,),
                target_repair_depth=target_depth,
                target_repair_depth_frozen=target_depth,
                canonical_actual_repair_count=0,
                branch_actual_repair_count=0,
            )
            trajectories.append(trajectory)
            return FakeBuildResult(
                task,
                root,
                tuple(trajectories),
                tuple(audit_records),
                tuple(candidates),
                source_decision,
                budget.assert_consistent(require_no_pending=True),
            )

        # B and C deliberately treat the initial natural path as failed. All
        # subsequent verifier/repair/regeneration material is audit-only.
        natural_failure = TrajectoryRecord(
            task_id=task.task_id,
            trajectory_id=trajectory_id,
            source_record_id=task.source_record_id,
            stage=task.stage,
            messages=natural_messages,
            images=task.images,
            image_content_hashes=task.image_content_hashes,
            canonical_source="natural",
            lineage_events=("natural",),
            root_snapshot_hash=root.snapshot_hash,
            start_snapshot_hash=root.snapshot_hash,
            final_answer_valid=False,
            step_validated=(False,),
            target_repair_depth=target_depth,
            target_repair_depth_frozen=target_depth,
            canonical_actual_repair_count=0,
            branch_actual_repair_count=0,
        )
        trajectories.append(natural_failure)

        verifier_result = self._request(
            budget, task, "verifier", parser=self._verifier_parser
        )
        verifier_text = str(verifier_result.value) if verifier_result.status == "successful" else ""
        repair_result = self._request(budget, task, "repair", parser=self._repair_parser)
        repair_text = str(repair_result.value) if repair_result.status == "successful" else ""
        regeneration_result = self._request(budget, task, "regeneration")
        regeneration_text = (
            str(regeneration_result.value) if regeneration_result.status == "successful" else ""
        )

        # Use a branch snapshot whose parent is the clean root. The branch's
        # records are explicitly contaminated and can never be exported.
        branch_snapshot = ImmutableSnapshot.fork(
            root,
            before_step_index=1,
            messages=list(natural_messages),
            original_images=list(task.image_refs),
            generation_boundaries=[{"boundary": "risk-branch", "step": 1}],
            sandbox_state_mode="stateless",
            base_sha=self.base_sha,
        )
        branch_id = f"{task.task_id}:controlled-1"
        verifier_unit = self._unit(
            task=task,
            trajectory_id=branch_id,
            suffix="verifier",
            supervision_type="verifier",
            context=natural_messages,
            target=verifier_text or "{}",
            evidence=("sandbox", "deterministic_rule"),
            controlled=True,
            repaired_error=True,
        )
        repair_unit = self._unit(
            task=task,
            trajectory_id=branch_id,
            suffix="repair",
            supervision_type="repair",
            context=natural_messages + ({"role": "assistant", "content": verifier_text},),
            target=repair_text or "{}",
            evidence=("sandbox", "deterministic_rule"),
            controlled=True,
            repaired_error=True,
        )
        regeneration_unit = self._unit(
            task=task,
            trajectory_id=branch_id,
            suffix="regeneration",
            supervision_type="regeneration",
            context=natural_messages
            + (
                {"role": "assistant", "content": verifier_text},
                {"role": "assistant", "content": repair_text},
            ),
            target=regeneration_text or "<think>failed regeneration</think>",
            evidence=("sandbox", "deterministic_rule"),
            controlled=True,
            regenerated=True,
        )
        audit_records.extend(
            [
                self._audit_record(
                    task,
                    f"{branch_id}:verifier",
                    "verifier",
                    verifier_unit.to_dict(),
                ),
                self._audit_record(
                    task,
                    f"{branch_id}:repair",
                    "repair",
                    repair_unit.to_dict(),
                ),
                self._audit_record(
                    task,
                    f"{branch_id}:regeneration",
                    "regeneration",
                    regeneration_unit.to_dict(),
                ),
            ]
        )
        controlled_trajectory = TrajectoryRecord(
            task_id=task.task_id,
            trajectory_id=branch_id,
            source_record_id=task.source_record_id,
            stage=task.stage,
            messages=natural_messages,
            images=task.images,
            image_content_hashes=task.image_content_hashes,
            canonical_source="controlled",
            lineage_events=("natural", "controlled", "verifier", "repair", "regeneration"),
            ancestor_trajectory_ids=(trajectory_id,),
            root_snapshot_hash=root.snapshot_hash,
            start_snapshot_hash=branch_snapshot.snapshot_hash,
            controlled_intervention=True,
            audit_context_received=True,
            final_answer_valid=True if task.task_id.endswith("B") else False,
            step_validated=(False,),
            repaired_error_steps=(0,),
            regenerated_steps=(0,),
            target_repair_depth=target_depth,
            target_repair_depth_frozen=target_depth,
            canonical_actual_repair_count=0,
            branch_actual_repair_count=1,
            supervision_units=(verifier_unit, repair_unit, regeneration_unit),
        )
        trajectories.append(controlled_trajectory)

        # Only B receives a clean replay. It shares the root state but has no
        # parent trajectory or controlled messages/context.
        if task.task_id.endswith("B"):
            replay_result = self._request(budget, task, "natural_replay")
            if replay_result.status == "successful":
                replay_text = str(replay_result.value)
                replay_messages = self._solver_messages(task, replay_text)
                replay = TrajectoryRecord(
                    task_id=task.task_id,
                    trajectory_id=f"{task.task_id}:natural-replay-2",
                    source_record_id=task.source_record_id,
                    stage=task.stage,
                    messages=replay_messages,
                    images=task.images,
                    image_content_hashes=task.image_content_hashes,
                    canonical_source="natural_replay",
                    lineage_events=("natural_replay",),
                    ancestor_trajectory_ids=(),
                    root_snapshot_hash=root.snapshot_hash,
                    start_snapshot_hash=root.snapshot_hash,
                    controlled_intervention=False,
                    audit_context_received=False,
                    final_answer_valid=True,
                    step_validated=(True,),
                    target_repair_depth=target_depth,
                    target_repair_depth_frozen=target_depth,
                    canonical_actual_repair_count=0,
                    branch_actual_repair_count=0,
                )
                trajectories.append(replay)

        return FakeBuildResult(
            task,
            root,
            tuple(trajectories),
            tuple(audit_records),
            tuple(candidates),
            source_decision,
            budget.assert_consistent(require_no_pending=True),
        )

    @staticmethod
    def _verifier_parser(value: str) -> dict[str, Any]:
        return parse_verifier_json(value)

    @staticmethod
    def _repair_parser(value: str) -> dict[str, Any]:
        return parse_repair_json(value)
