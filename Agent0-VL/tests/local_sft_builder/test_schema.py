from __future__ import annotations

import copy

import pytest

from tools.local_sft_builder.schema import (
    MigrationConflictError,
    SCHEMA_VERSION,
    SupervisionUnit,
    migrate_v1_to_v2,
)


def test_v1_record_migrates_to_v2_without_mutating_input() -> None:
    source = {
        "schema_version": "agent0vl.dataset.v1",
        "record_id": "r1",
        "record_type": "repair",
        "assistant_target": "{}",
        "nested": {"items": [1, 2]},
    }
    original = copy.deepcopy(source)

    migrated = migrate_v1_to_v2(source)

    assert source == original
    assert migrated["schema_version"] == SCHEMA_VERSION
    assert migrated["supervision_type"] == "repair"
    assert "record_type" not in migrated
    assert migrated["migrated_from_schema_version"] == "agent0vl.dataset.v1"


def test_v1_to_v2_migration_is_idempotent() -> None:
    source = {"schema_version": "agent0vl.dataset.v1", "record_type": "verifier"}
    once = migrate_v1_to_v2(source)
    twice = migrate_v1_to_v2(once)
    assert twice == once


def test_v2_record_is_not_marked_as_migrated() -> None:
    source = {"schema_version": SCHEMA_VERSION, "supervision_type": "repair"}
    migrated = migrate_v1_to_v2(source)
    assert migrated == source


def test_equal_legacy_and_v2_fields_are_normalized() -> None:
    source = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "verifier",
        "supervision_type": "verifier",
    }
    migrated = migrate_v1_to_v2(source)
    assert migrated["supervision_type"] == "verifier"
    assert "record_type" not in migrated
    assert "migrated_from_schema_version" not in migrated


def test_conflicting_supervision_fields_require_review() -> None:
    with pytest.raises(MigrationConflictError):
        migrate_v1_to_v2(
            {
                "schema_version": "agent0vl.dataset.v1",
                "record_type": "repair",
                "supervision_type": "verifier",
            }
        )


def test_supervision_unit_has_one_target_and_one_type() -> None:
    unit = SupervisionUnit(
        record_id="u1",
        task_id="t1",
        trajectory_id="tr1",
        supervision_type="solver_positive",
        context_messages=(
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "history"},
        ),
        assistant_target="target",
    )
    output = unit.to_dict()
    assert output["supervision_type"] == "solver_positive"
    assert "record_type" not in output
    assert output["assistant_target"] == "target"
