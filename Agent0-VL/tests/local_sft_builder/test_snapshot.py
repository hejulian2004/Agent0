from __future__ import annotations

import pytest

from tools.local_sft_builder.snapshots import ImmutableSnapshot


BASE_SHA = "a" * 64


def _root() -> ImmutableSnapshot:
    return ImmutableSnapshot.create(
        messages=[{"role": "user", "content": "Q"}],
        original_images=[
            {"asset_id": "dataset/item-1/image-0", "content_sha256": "b" * 64}
        ],
        observation_events=[],
        generation_boundaries=[{"boundary": "root", "step": 0}],
        sandbox_state_mode="stateless",
        base_sha=BASE_SHA,
    )


def test_snapshot_is_canonical_content_addressed_and_immutable() -> None:
    first = _root()
    second = _root()
    assert first.snapshot_id == second.snapshot_id
    assert first.snapshot_hash == second.snapshot_hash
    assert first.snapshot_id == "snap_" + first.snapshot_hash

    exposed = first.payload
    exposed["messages"][0]["content"] = "mutated"
    exposed["original_images"][0]["asset_id"] = "other"
    assert first.payload["messages"][0]["content"] == "Q"
    assert first.payload["original_images"][0]["asset_id"] == "dataset/item-1/image-0"
    assert first.snapshot_hash == second.snapshot_hash


def test_stateless_sandbox_has_no_fake_state_hash() -> None:
    snapshot = _root()
    assert snapshot.payload["sandbox_state_mode"] == "stateless"
    assert snapshot.payload["sandbox_state_hash"] is None
    with pytest.raises(ValueError):
        ImmutableSnapshot.create(
            messages=[{"role": "user", "content": "Q"}],
            sandbox_state_mode="stateless",
            sandbox_state_hash="b" * 64,
        )


def test_serialized_sandbox_requires_a_deterministic_hash() -> None:
    with pytest.raises(ValueError):
        ImmutableSnapshot.create(
            messages=[{"role": "user", "content": "Q"}],
            sandbox_state_mode="serialized",
        )
    snapshot = ImmutableSnapshot.create(
        messages=[{"role": "user", "content": "Q"}],
        sandbox_state_mode="serialized",
        sandbox_state_hash="b" * 64,
    )
    assert snapshot.payload["sandbox_state_hash"] == "b" * 64


def test_branch_references_parent_and_preserves_branch_start_state() -> None:
    root = _root()
    branch = ImmutableSnapshot.fork(
        root,
        before_step_index=1,
        messages=root.payload["messages"] + [{"role": "assistant", "content": "branch"}],
        original_images=root.payload["original_images"],
        generation_boundaries=[{"boundary": "risk", "step": 1}],
        sandbox_state_mode="stateless",
        base_sha=BASE_SHA,
    )
    assert branch.parent_snapshot_id == root.snapshot_id
    assert branch.parent_snapshot_hash == root.snapshot_hash
    assert branch.branch_start_matches_parent(root)
    assert branch.payload["branch_start_state_hash"] == root.state_hash


def test_absolute_paths_are_not_valid_snapshot_asset_ids() -> None:
    with pytest.raises(ValueError):
        ImmutableSnapshot.create(
            messages=[{"role": "user", "content": "Q"}],
            original_images=[{"asset_id": r"D:\data\image.png", "content_sha256": "b" * 64}],
        )


@pytest.mark.parametrize("field_name", ("original_images", "derived_images"))
def test_snapshot_assets_require_hashed_mappings(field_name: str) -> None:
    invalid_assets = (
        "dataset/item-1/image-0",
        {"asset_id": "dataset/item-1/image-0"},
        {"asset_id": "dataset/item-1/image-0", "content_sha256": None},
        {"asset_id": "dataset/item-1/image-0", "content_sha256": "A" * 64},
        {"asset_id": "dataset/item-1/image-0", "content_sha256": "b" * 63},
    )
    for asset in invalid_assets:
        with pytest.raises((TypeError, ValueError)):
            ImmutableSnapshot.create(
                messages=[{"role": "user", "content": "Q"}],
                **{field_name: [asset]},
            )


@pytest.mark.parametrize("field_name", ("original_images", "derived_images"))
def test_snapshot_hash_binds_asset_content_hash(field_name: str) -> None:
    first = ImmutableSnapshot.create(
        messages=[{"role": "user", "content": "Q"}],
        **{
            field_name: [
                {"asset_id": "dataset/item-1/image-0", "content_sha256": "b" * 64}
            ]
        },
    )
    second = ImmutableSnapshot.create(
        messages=[{"role": "user", "content": "Q"}],
        **{
            field_name: [
                {"asset_id": "dataset/item-1/image-0", "content_sha256": "c" * 64}
            ]
        },
    )
    assert first.snapshot_hash != second.snapshot_hash
