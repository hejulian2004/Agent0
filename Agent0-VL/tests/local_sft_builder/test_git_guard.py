from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.local_sft_builder.git_guard import (
    ALLOWED_EXACT_PATHS,
    ALLOWED_MODIFICATION_PATHS,
    UpstreamImmutabilityError,
    assert_upstream_immutable,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def git_guard_change(status: str, path: str, old_path: str | None = None):
    """Build a ``GitChange`` for the parametrized guard tests."""

    from tools.local_sft_builder.git_guard import GitChange

    return GitChange(status, path, old_path)


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


def test_only_root_agents_file_is_an_exact_exception() -> None:
    assert ALLOWED_EXACT_PATHS == {"AGENTS.md"}


def test_only_root_gitignore_may_be_modified() -> None:
    assert ALLOWED_MODIFICATION_PATHS == {".gitignore"}


def test_root_gitignore_modification_is_tolerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.local_sft_builder import git_guard

    monkeypatch.setattr(
        git_guard,
        "changed_paths",
        lambda repo_root, base_sha, head_sha="HEAD": [
            git_guard.GitChange("M", ".gitignore"),
            git_guard.GitChange(
                "A", "Agent0-VL/tools/local_sft_builder/new_module.py"
            ),
        ],
    )
    changes = assert_upstream_immutable(REPO_ROOT, "base")
    assert [change.path for change in changes] == [
        ".gitignore",
        "Agent0-VL/tools/local_sft_builder/new_module.py",
    ]


@pytest.mark.parametrize(
    "change",
    [
        # A deletion is never allowed, even of the one modifiable path.
        git_guard_change("D", ".gitignore"),
        # A rename away from it is never allowed either.
        git_guard_change("R100", "README.md", ".gitignore"),
        # The exception must not leak to any other config file.
        git_guard_change("M", "Agent0-VL/.gitignore"),
        git_guard_change("M", "AGENTS.md"),
        git_guard_change("M", ".workbuddy-ai/MEMORY.md"),
    ],
)
def test_gitignore_exception_does_not_leak(
    monkeypatch: pytest.MonkeyPatch, change
) -> None:
    from tools.local_sft_builder import git_guard

    monkeypatch.setattr(
        git_guard,
        "changed_paths",
        lambda repo_root, base_sha, head_sha="HEAD": [change],
    )
    with pytest.raises(UpstreamImmutabilityError):
        assert_upstream_immutable(REPO_ROOT, "base")
