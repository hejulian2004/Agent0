from __future__ import annotations

import pytest

from tools.sft_builder.sample_sft import (
    _mulberry_source,
    _normalize_form_feed,
    _select_rows,
)


def _row(content: str, *, repair: bool = False) -> dict:
    messages = [{"role": "user", "content": content}]
    if repair:
        messages.append({
            "role": "user",
            "content": "Now switch to the Self-Repair role.",
        })
    return {"messages": messages, "images": []}


def test_mulberry_source_normalizes_case_and_cauldron_wrapper() -> None:
    row = {
        "images": [
            "/data/mulberry_images/cauldron/DVQA/images/example.png"
        ]
    }
    assert _mulberry_source(row) == "mulberry:dvqa"


def test_form_feed_normalization_preserves_page_break_as_newline() -> None:
    row = {"messages": [{"role": "user", "content": "before\fafter"}]}
    assert _normalize_form_feed(row)
    assert row["messages"][0]["content"] == "before\nafter"


def test_selection_keeps_all_repairs_and_covers_each_source() -> None:
    buckets = {
        "legacy": [("legacy-1", _row("legacy 1")), ("legacy-2", _row("legacy 2"))],
        "mulberry:chartqa": [
            ("repair-1", _row("repair", repair=True)),
            ("chart-2", _row("chart 2")),
        ],
        "mulberry:vizwiz": [("vizwiz-1", _row("vizwiz"))],
    }

    selected = _select_rows(
        buckets,
        target=3,
        seed=17,
        require_all_repairs=True,
        ensure_source_coverage=True,
    )

    assert len(selected) == 3
    assert {source for source, _ in selected} == set(buckets)
    assert sum(
        any("Self-Repair role" in message["content"] for message in row["messages"])
        for _, row in selected
    ) == 1


def test_selection_rejects_required_repairs_exceeding_target() -> None:
    buckets = {
        "mulberry:a": [
            ("repair-1", _row("repair 1", repair=True)),
            ("repair-2", _row("repair 2", repair=True)),
        ]
    }

    with pytest.raises(RuntimeError, match="Required repair rows"):
        _select_rows(
            buckets,
            target=1,
            seed=17,
            require_all_repairs=True,
            ensure_source_coverage=True,
        )
