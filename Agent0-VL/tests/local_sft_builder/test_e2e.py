from __future__ import annotations

import json

from tools.local_sft_builder.fake_builder import FakeTrajectoryBuilder, SourceTask
from tools.local_sft_builder.pipeline import export_pipeline_result, validate_project_deduplicate
from tools.local_sft_builder.projector import Projector
from tools.local_sft_builder.runtime import ScriptedBackend
from tools.local_sft_builder.validator import validate_supervision_unit, validate_trajectory_for_export


BASE_SHA = "a" * 64
IMAGE_A_HASH = "b" * 64
IMAGE_B_HASH = "c" * 64


def _final(answer: str) -> str:
    return f"<think>deterministic fixture reasoning</think>\nCONFIDENCE: 0.95\nFINAL_ANSWER: {answer}"


def _tasks() -> tuple[SourceTask, ...]:
    return (
        SourceTask(
            task_id="task-A",
            source_record_id="source-A",
            source_dataset="fixture",
            stage="sft_stage1",
            usage_partition="sft_stage1",
            question="<image>\nWhat is 2 + 2?",
            images=("fixture/task-A/image-0.png",),
            image_refs=(
                {
                    "asset_id": "fixture/task-A/image-0",
                    "content_sha256": IMAGE_A_HASH,
                },
            ),
            ground_truth="4",
        ),
        SourceTask(
            task_id="task-B",
            source_record_id="source-B",
            source_dataset="fixture",
            stage="sft_stage2",
            usage_partition="sft_stage2",
            question="<image>\nUse Python to calculate 2 + 2.",
            images=("fixture/task-B/image-0.png",),
            image_refs=(
                {
                    "asset_id": "fixture/task-B/image-0",
                    "content_sha256": IMAGE_B_HASH,
                },
            ),
            ground_truth="4",
        ),
        SourceTask(
            task_id="task-C",
            source_record_id="source-C",
            source_dataset="fixture",
            stage="sft_stage1",
            usage_partition="sft_stage1",
            question="This natural rollout is intentionally incomplete.",
            ground_truth="4",
        ),
    )


def _backend() -> ScriptedBackend:
    natural_failure = "<think>an incomplete natural attempt"
    return ScriptedBackend(
        {
            "task-A:task_analysis": [json.dumps({"target_repair_depth": 0})],
            "task-A:natural": [_final("4")],
            "task-B:task_analysis": [json.dumps({"target_repair_depth": 1})],
            "task-B:natural": [
                "<think>The intermediate claim is intentionally wrong.</think>\n```python\nprint(2 + 2)\n```"
            ],
            "task-B:verifier": [
                '{"step_index":0,"score":-0.9,"confidence":0.95,"critique":"The claim is wrong.","tool_check":true}'
            ],
            "task-B:repair": [
                '{"action":"PATCH","target_step":0,"patch_type":"text","new_content":"Use 4."}'
            ],
            "task-B:regeneration": [_final("4")],
            "task-B:natural_replay": [_final("4")],
            "task-C:task_analysis": [json.dumps({"target_repair_depth": 1})],
            "task-C:natural": [natural_failure],
            "task-C:verifier": [
                '{"step_index":0,"score":-1.0,"confidence":0.9,"critique":"Incomplete.","tool_check":false}'
            ],
            "task-C:repair": [
                '{"action":"PATCH","target_step":0,"patch_type":"text","new_content":"Complete it."}'
            ],
            "task-C:regeneration": [_final("4")],
        }
    )


def test_fake_e2e_a_b_c_preserves_lineage_boundary_and_projection(tmp_path) -> None:
    backend = _backend()
    builder = FakeTrajectoryBuilder(
        backend=backend,
        budget_db=str(tmp_path / "teacher.sqlite3"),
        base_sha=BASE_SHA,
    )
    results = [builder.build_task(task) for task in _tasks()]

    a, b, c = results
    assert a.trajectories[0].image_content_hashes == (IMAGE_A_HASH,)
    assert all(
        trajectory.image_content_hashes == (IMAGE_B_HASH,)
        for trajectory in b.trajectories
    )
    assert all(not trajectory.image_content_hashes for trajectory in c.trajectories)

    a_decision, a_candidate = validate_trajectory_for_export(
        a.trajectories[0], source_guard=a.source_guard
    )
    assert a_decision.exportable and a_candidate is not None

    replay = next(item for item in b.trajectories if item.canonical_source == "natural_replay")
    b_decision, b_candidate = validate_trajectory_for_export(
        replay, source_guard=b.source_guard
    )
    assert b_decision.exportable and b_candidate is not None
    assert replay.ancestor_trajectory_ids == ()
    assert replay.root_snapshot_hash == b.root_snapshot.snapshot_hash
    assert not replay.has_controlled_ancestry
    assert all("repair" not in item["content"].lower() for item in replay.messages)

    controlled = next(item for item in b.trajectories if item.canonical_source == "controlled")
    controlled_decision, controlled_candidate = validate_trajectory_for_export(
        controlled, source_guard=b.source_guard
    )
    assert not controlled_decision.exportable
    assert controlled_candidate is None
    assert controlled.has_controlled_ancestry
    assert all(validate_supervision_unit(unit).status == "accepted"
               for unit in controlled.supervision_units)

    assert len(c.trajectories) == 2
    assert not any(item.canonical_source == "natural_replay" for item in c.trajectories)
    c_candidates = [
        validate_trajectory_for_export(item, source_guard=c.source_guard)[1]
        for item in c.trajectories
    ]
    assert all(candidate is None for candidate in c_candidates)

    rows = Projector().project_many([a_candidate, b_candidate])
    assert len(rows) == 2
    assert {row.image_content_hashes for row in rows} == {
        (IMAGE_A_HASH,),
        (IMAGE_B_HASH,),
    }
    assert all(set(row.to_dict()) == {"messages", "images"} for row in rows)

    pipeline = validate_project_deduplicate(
        [trajectory for result in results for trajectory in result.trajectories],
        {result.task.task_id: result.source_guard for result in results},
    )
    assert len(pipeline.rows_before_dedup) == 2
    assert len(pipeline.rows_after_dedup) == 2
    output_path = tmp_path / "final.jsonl"
    assert export_pipeline_result(pipeline, output_path) == 2
    serialized = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert all(set(row) == {"messages", "images"} for row in serialized)

    for result in results:
        assert result.budget_stats.pending == 0
        assert result.budget_stats.consumed_slot <= 32


def test_invalid_image_hash_stops_before_snapshot_or_generation(tmp_path) -> None:
    task = SourceTask(
        task_id="task-invalid-image",
        source_record_id="source-invalid-image",
        source_dataset="fixture",
        stage="sft_stage1",
        usage_partition="sft_stage1",
        question="<image>\nThis image hash is invalid.",
        images=("fixture/invalid/image-0.png",),
        image_refs=(
            {
                "asset_id": "fixture/invalid/image-0",
                "content_sha256": None,
            },
        ),
    )
    builder = FakeTrajectoryBuilder(
        backend=ScriptedBackend({}),
        budget_db=str(tmp_path / "invalid-image.sqlite3"),
        base_sha=BASE_SHA,
    )

    result = builder.build_task(task)

    assert result.source_guard.status == "review_required"
    assert result.root_snapshot is None
    assert result.trajectories == ()
    assert len(result.audit_records) == 1
    assert result.budget_stats.consumed_slot == 0
