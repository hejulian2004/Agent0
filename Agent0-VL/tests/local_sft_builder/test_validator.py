from __future__ import annotations

from tools.local_sft_builder.schema import SupervisionUnit, TrajectoryRecord
from tools.local_sft_builder.snapshots import ImmutableSnapshot
from tools.local_sft_builder.source_guard import SourceLeakageGuard
from tools.local_sft_builder.validator import (
    validate_supervision_unit,
    validate_target_repair_depth,
    validate_trajectory_for_export,
)


def _source_decision():
    return SourceLeakageGuard().check(
        {
            "task_id": "t1",
            "source_record_id": "o1",
            "original_id": "o1",
            "source_dataset": "fixture",
            "source_revision": "v1",
            "question": "Q",
            "split": "train",
            "usage_partition": "sft_stage1",
        },
        expected_stage="sft_stage1",
    )


def _root() -> ImmutableSnapshot:
    return ImmutableSnapshot.create(messages=[{"role": "user", "content": "Q"}])


def _trajectory(**overrides) -> TrajectoryRecord:
    root = _root()
    values = {
        "task_id": "t1",
        "trajectory_id": "tr1",
        "source_record_id": "o1",
        "stage": "sft_stage1",
        "messages": (
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": "<think>done</think>\nCONFIDENCE: 0.9\nFINAL_ANSWER: 4",
            },
        ),
        "root_snapshot_hash": root.snapshot_hash,
        "start_snapshot_hash": root.snapshot_hash,
        "final_answer_valid": True,
        "step_validated": (True,),
        "target_repair_depth": 0,
        "target_repair_depth_frozen": 0,
    }
    values.update(overrides)
    return TrajectoryRecord(**values)


def test_clean_natural_trajectory_is_export_eligible() -> None:
    decision, candidate = validate_trajectory_for_export(
        _trajectory(), source_guard=_source_decision()
    )
    assert decision.status == "accepted"
    assert decision.supervision_type == "solver_positive"
    assert decision.exportable
    assert candidate is not None


def test_controlled_ancestry_and_repaired_steps_are_not_positive() -> None:
    for overrides in (
        {"controlled_intervention": True, "lineage_events": ("controlled",)},
        {"repaired_error_steps": (0,)},
        {"regenerated_steps": (0,)},
    ):
        decision, candidate = validate_trajectory_for_export(
            _trajectory(**overrides), source_guard=_source_decision()
        )
        assert decision.status in {"rejected", "review_required"}
        assert candidate is None


def test_target_repair_depth_is_immutable() -> None:
    record = _trajectory(target_repair_depth=1, target_repair_depth_frozen=0)
    try:
        validate_target_repair_depth(record)
    except ValueError as exc:
        assert "changed after Task Analysis" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mutable target depth was accepted")


def test_audit_units_need_independent_local_evidence() -> None:
    base = dict(
        record_id="u1",
        task_id="t1",
        trajectory_id="tr1",
        context_messages=({"role": "user", "content": "Q"},),
        assistant_target="{}",
    )
    no_evidence = validate_supervision_unit(
        SupervisionUnit(supervision_type="repair", **base)
    )
    with_evidence = validate_supervision_unit(
        SupervisionUnit(
            supervision_type="repair",
            evidence_sources=("ground_truth", "sandbox"),
            **base,
        )
    )
    assert no_evidence.status == "review_required"
    assert "missing_independent_local_evidence" in no_evidence.reasons
    assert with_evidence.status == "accepted"
    assert not with_evidence.exportable


def test_regeneration_unit_is_audit_valid_but_not_solver_positive() -> None:
    unit = SupervisionUnit(
        record_id="u2",
        task_id="t1",
        trajectory_id="tr1",
        supervision_type="regeneration",
        context_messages=({"role": "user", "content": "Q"},),
        assistant_target="<think>corrected</think>",
        evidence_sources=("numeric_recomputation",),
        regenerated=True,
    )
    decision = validate_supervision_unit(unit)
    assert decision.status == "accepted"
    assert not decision.solver_positive_eligible
    assert not decision.exportable
