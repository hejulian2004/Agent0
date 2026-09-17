from __future__ import annotations

import json

import pytest

from tools.local_sft_builder.swift_loader_smoke import (
    SwiftLoaderUnavailable,
    load_with_ms_swift,
)


def test_ms_swift_loader_smoke_uses_real_preprocessing_path(tmp_path) -> None:
    dataset_path = tmp_path / "rows.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ],
                "images": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        train_dataset, validation_dataset = load_with_ms_swift(dataset_path)
    except SwiftLoaderUnavailable:
        pytest.skip("target environment has no ms-swift; run this in the training venv")
    assert len(train_dataset) == 1
    assert validation_dataset is None or len(validation_dataset) == 0
