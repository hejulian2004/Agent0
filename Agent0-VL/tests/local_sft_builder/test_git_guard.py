from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.local_sft_builder.git_guard import (
    UpstreamImmutabilityError,
    assert_upstream_immutable,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT}", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def test_current_dev_commit_passes_additive_allowlist() -> None:
    base_sha = _git("rev-parse", "main")
    changes = assert_upstream_immutable(REPO_ROOT, base_sha)
    assert changes
    assert all(change.status == "A" for change in changes)


def test_upstream_modification_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.local_sft_builder import git_guard

    monkeypatch.setattr(
        git_guard,
        "changed_paths",
        lambda repo_root, base_sha, head_sha="HEAD": [
            git_guard.GitChange("M", "Agent0-VL/scripts/prompt.txt")
        ],
    )
    with pytest.raises(UpstreamImmutabilityError):
        assert_upstream_immutable(REPO_ROOT, "base")
