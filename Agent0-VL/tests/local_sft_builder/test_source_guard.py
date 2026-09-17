from __future__ import annotations

from tools.local_sft_builder.source_guard import SourceLeakageGuard


def _row(**overrides):
    row = {
        "task_id": "task-1",
        "source_record_id": "original-1",
        "original_id": "original-1",
        "source_dataset": "fixture",
        "source_revision": "v1",
        "question": "2+2",
        "split": "train",
        "usage_partition": "sft_stage1",
        "image_content_hashes": [],
    }
    row.update(overrides)
    return row


def test_source_guard_rejects_forbidden_identity_before_generation() -> None:
    guard = SourceLeakageGuard([_row(split="test", usage_partition="eval")])
    decision = guard.check(_row(), expected_stage="sft_stage1")
    assert decision.status == "rejected"
    assert "forbidden_source_identity_overlap" in decision.reasons


def test_source_guard_fails_closed_on_unknown_split() -> None:
    decision = SourceLeakageGuard().check(_row(split=None), expected_stage="sft_stage1")
    assert decision.status == "review_required"


def test_source_guard_fails_closed_on_missing_source_revision() -> None:
    for revision in (None, "   "):
        decision = SourceLeakageGuard().check(
            _row(source_revision=revision), expected_stage="sft_stage1"
        )
        assert decision.status == "review_required"
        assert any("source_revision" in reason for reason in decision.reasons)


def test_source_guard_fails_closed_on_incomplete_or_noncanonical_image_hashes() -> None:
    cases = (
        {"images": ["image.png"], "image_content_hashes": []},
        {"images": ["image.png"], "image_content_hashes": [None]},
        {"images": ["image.png"], "image_content_hashes": ["A" * 64]},
        {"images": ["image.png"], "image_content_hashes": ["a" * 63]},
        {"source_content_hash": "A" * 64},
    )
    for overrides in cases:
        decision = SourceLeakageGuard().check(
            _row(**overrides), expected_stage="sft_stage1"
        )
        assert decision.status == "review_required"
        assert any("invalid_source_identity" in reason for reason in decision.reasons)


def test_source_guard_accepts_complete_visual_identity() -> None:
    decision = SourceLeakageGuard().check(
        _row(images=["image.png"], image_content_hashes=["a" * 64]),
        expected_stage="sft_stage1",
    )
    assert decision.accepted
    assert decision.identity is not None
    assert ("image_content_hash", "a" * 64) in decision.identity.keys()


def test_source_guard_rejects_non_train_and_stage_mismatch() -> None:
    guard = SourceLeakageGuard()
    assert guard.check(_row(split="validation"), expected_stage="sft_stage1").status == "rejected"
    assert guard.check(_row(usage_partition="rl"), expected_stage="sft_stage1").status == "rejected"
    assert guard.check(_row(usage_partition="sft_stage2"), expected_stage="sft_stage1").status == "rejected"


def test_source_guard_accepts_train_sft_partition() -> None:
    decision = SourceLeakageGuard().check(_row(), expected_stage="sft_stage1")
    assert decision.accepted
    assert decision.identity is not None
