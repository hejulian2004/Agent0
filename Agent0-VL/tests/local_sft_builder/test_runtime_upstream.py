"""Tests for the frozen-upstream sandbox integration.

These exercise the local CPU sandbox only; no dataset or Teacher is involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.local_sft_builder.runtime import (
    RealSandboxRuntime,
    UpstreamSandboxRunner,
    _agent0_vl_root,
)


@pytest.fixture(scope="module")
def runner() -> UpstreamSandboxRunner:
    return UpstreamSandboxRunner(timeout_seconds=10.0)


def test_agent0_vl_root_is_the_package_parent() -> None:
    root = _agent0_vl_root()

    assert (root / "sandbox").is_dir()
    assert str(root) in sys.path


def test_successful_execution_matches_upstream_shape(
    runner: UpstreamSandboxRunner,
) -> None:
    result = runner.run("print(40 + 2)")

    assert result.status == "success"
    assert result.stdout.strip() == "42"
    assert result.stderr == ""
    assert set(result.to_dict()) == {"status", "run_result", "returncode"}


def test_runtime_error_reports_stderr(runner: UpstreamSandboxRunner) -> None:
    result = runner.run("raise ValueError('boom')")

    assert result.status == "runtime_error"
    assert "boom" in result.stderr


def test_timeout_is_distinguishable_from_failure() -> None:
    slow = UpstreamSandboxRunner(timeout_seconds=1.0)
    result = slow.run("import time\ntime.sleep(30)\nprint('never')")

    assert result.status == "timeout"
    assert result.stdout == ""


def test_no_tool_answer_produces_no_observation() -> None:
    runtime = RealSandboxRuntime(sandbox=UpstreamSandboxRunner(timeout_seconds=10.0))
    solver_text = "<think>already known</think>\nCONFIDENCE: 0.9\nFINAL_ANSWER: 42"

    calls, results, observation = runtime.execute_solver_text(solver_text)

    assert calls == ()
    assert results == []
    assert observation is None


def test_observation_wrapper_is_source_compatible() -> None:
    runtime = RealSandboxRuntime(sandbox=UpstreamSandboxRunner(timeout_seconds=10.0))
    solver_text = "<think>compute</think>\n```python\nprint(2 + 2)\n```"

    calls, results, observation = runtime.execute_solver_text(solver_text)

    assert [call.tool_name for call in calls] == ["PythonExec"]
    assert calls[0].code.strip() == "print(2 + 2)"
    assert len(results) == 1
    assert observation == "\n[Code Execution Result]\nOutput: 4\n\n"


def test_multiple_sequential_blocks_each_execute() -> None:
    runtime = RealSandboxRuntime(sandbox=UpstreamSandboxRunner(timeout_seconds=10.0))
    solver_text = (
        "<think>two steps</think>\n"
        "```python\nprint('first')\n```\n"
        "```python\nprint('second')\n```"
    )

    calls, results, observation = runtime.execute_solver_text(solver_text)

    assert len(calls) == 2
    assert len(results) == 2
    assert results[0]["run_result"]["stdout"].strip() == "first"
    assert results[1]["run_result"]["stdout"].strip() == "second"
    assert observation is not None
    assert "Output: first" in observation
    assert "Output: second" in observation


def test_execute_as_message_uses_user_role() -> None:
    runtime = RealSandboxRuntime(sandbox=UpstreamSandboxRunner(timeout_seconds=10.0))

    _calls, _results, message = runtime.execute_as_message(
        "<think>compute</think>\n```python\nprint(1)\n```"
    )

    assert message is not None
    assert message["role"] == "user"
    assert message["content"].startswith("\n[Code Execution Result]\n")


def test_json_tool_call_shape_is_still_rejected() -> None:
    runtime = RealSandboxRuntime(sandbox=UpstreamSandboxRunner(timeout_seconds=10.0))

    with pytest.raises(Exception):
        runtime.execute_solver_text('{"tool_name": "PythonExec", "tool_input": "1+1"}')


def test_describe_is_manifest_safe() -> None:
    from tools.local_sft_builder.manifest import build_manifest

    runner = UpstreamSandboxRunner(timeout_seconds=10.0)
    manifest = build_manifest(
        run_id="run-1",
        base_sha="a" * 40,
        extra=runner.describe(),
    )

    assert manifest["sandbox_backend"] == "upstream_agent0_vl_sandbox"
    assert manifest["sandbox_timeout_seconds"] == 10.0


def test_missing_sandbox_package_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from tools.local_sft_builder.runtime import UpstreamSandboxUnavailable

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("sandbox"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    with pytest.raises(UpstreamSandboxUnavailable):
        UpstreamSandboxRunner()
