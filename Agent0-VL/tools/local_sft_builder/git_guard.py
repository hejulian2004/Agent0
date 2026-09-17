"""Mechanical guard for the additive-only upstream policy."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


ALLOWED_ADDITION_PREFIXES = (
    "Agent0-VL/tools/local_sft_builder/",
    "Agent0-VL/tests/local_sft_builder/",
)
ALLOWED_EXACT_PATHS = frozenset({"AGENTS.md"})

# The single deliberate exception to "additions only".
#
# The repo-root .gitignore is build configuration, not upstream source: it
# cannot change what the frozen Agent0-VL code does. It is listed here so that
# the builder's own workspace directory (``.workbuddy-ai/``, which holds local
# agent memory) can be ignored -- otherwise ``git add -A`` drags that local
# state into a commit and trips this very guard.
#
# The exception is intentionally narrower than the addition allowlist: only a
# *modification* of exactly these paths is tolerated. Deletions, renames and
# every other upstream modification still fail, and no other file may be
# touched at all.
ALLOWED_MODIFICATION_PATHS = frozenset({".gitignore"})


@dataclass(frozen=True)
class GitChange:
    status: str
    path: str
    old_path: str | None = None


class UpstreamImmutabilityError(RuntimeError):
    pass


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo_root}", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout


def changed_paths(repo_root: str | Path, base_sha: str, head_sha: str = "HEAD") -> list[GitChange]:
    output = _git(Path(repo_root), "diff", "--name-status", "--find-renames", f"{base_sha}..{head_sha}")
    changes: list[GitChange] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") and len(fields) >= 3:
            changes.append(GitChange(status, fields[2], fields[1]))
        else:
            changes.append(GitChange(status, fields[-1]))
    return changes


def assert_upstream_immutable(
    repo_root: str | Path,
    base_sha: str,
    head_sha: str = "HEAD",
) -> list[GitChange]:
    changes = changed_paths(repo_root, base_sha, head_sha)
    violations: list[GitChange] = []
    for change in changes:
        allowed_addition = change.status.startswith("A") and (
            change.path in ALLOWED_EXACT_PATHS
            or any(change.path.startswith(prefix) for prefix in ALLOWED_ADDITION_PREFIXES)
        )
        allowed_modification = (
            change.status == "M" and change.path in ALLOWED_MODIFICATION_PATHS
        )
        if not (allowed_addition or allowed_modification):
            violations.append(change)
    if violations:
        rendered = ", ".join(f"{item.status}:{item.path}" for item in violations)
        raise UpstreamImmutabilityError(f"upstream/addition allowlist violation: {rendered}")
    return changes
