from __future__ import annotations

import pytest

from tools.sft_builder.merge_sft_stages import _merge_rows


def _row(text: str, *, repair: bool = False) -> dict:
    messages = [{"role": "user", "content": text}]
    if repair:
        messages.append({
            "role": "user",
            "content": "Now switch to the Self-Repair role.",
        })
    return {"messages": messages, "images": []}


def test_merge_preserves_stage_order_and_requires_repairs() -> None:
    stage1 = [_row("stage1-a"), _row("stage1-repair", repair=True)]
    stage2 = [_row("stage2-a")]

    merged = _merge_rows(
        stage1,
        stage2,
        expected_stage1_rows=2,
        expected_stage2_rows=1,
        expected_stage1_repairs=1,
    )

    assert merged == [*stage1, *stage2]


def test_merge_rejects_cross_stage_duplicates() -> None:
    duplicate = _row("same payload")

    with pytest.raises(ValueError, match="Exact duplicate"):
        _merge_rows(
            [duplicate],
            [duplicate],
            expected_stage1_rows=1,
            expected_stage2_rows=1,
            expected_stage1_repairs=0,
        )


def test_merge_rejects_wrong_repair_count() -> None:
    with pytest.raises(ValueError, match="Repair rows"):
        _merge_rows(
            [_row("no repair")],
            [_row("stage2")],
            expected_stage1_rows=1,
            expected_stage2_rows=1,
            expected_stage1_repairs=1,
        )
