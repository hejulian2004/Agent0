"""Compatibility smoke through ms-swift's real dataset preprocessing path."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SwiftLoaderUnavailable(RuntimeError):
    pass


def load_with_ms_swift(path: str | Path) -> tuple[Any, Any]:
    """Load a local JSONL through ``swift.dataset.load_dataset``.

    This deliberately stops after preprocessing/loading. It does not invoke
    ``swift sft`` and never starts model training.
    """

    try:
        from swift.dataset import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on target env
        raise SwiftLoaderUnavailable("ms-swift is not installed") from exc
    train_dataset, validation_dataset = load_dataset(
        str(path),
        split_dataset_ratio=0,
        shuffle=False,
        streaming=False,
        load_from_cache_file=False,
    )
    if len(train_dataset):
        first = train_dataset[0]
        if not isinstance(first, dict) or "messages" not in first:
            raise ValueError("ms-swift preprocessing did not retain messages")
        if "images" in first and first["images"] is not None and not isinstance(
            first["images"], (list, tuple)
        ):
            raise ValueError("ms-swift images field is not a sequence")
    return train_dataset, validation_dataset
