"""Runtime adapters used by the local builder.

``DeterministicPythonSandbox`` is a small bounded subprocess used by the fake
builder and its tests.  ``UpstreamSandboxRunner`` routes real tool execution
through the frozen upstream sandbox so that observations match the rollout
runtime byte for byte.  Importing the upstream package never modifies it.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .protocol import (
    PythonToolCall,
    extract_python_blocks,
    format_code_execution_observation,
    observation_message,
    reject_json_tool_call,
)


@dataclass(frozen=True)
class SandboxResult:
    status: str
    stdout: str
    stderr: str
    returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_result": {"stdout": self.stdout, "stderr": self.stderr},
            "returncode": self.returncode,
        }


class DeterministicPythonSandbox:
    """Execute Python through a bounded subprocess with deterministic capture."""

    def __init__(self, *, timeout_seconds: float = 5.0, max_output_chars: int = 512):
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run(self, code: str) -> SandboxResult:
        if not isinstance(code, str):
            raise TypeError("sandbox code must be text")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                input="",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                status="timeout",
                stdout=(exc.stdout or "")[: self.max_output_chars],
                stderr=(exc.stderr or "")[: self.max_output_chars],
            )
        except OSError as exc:
            return SandboxResult(status="runtime_error", stdout="", stderr=str(exc))

        return SandboxResult(
            status="success" if completed.returncode == 0 else "runtime_error",
            stdout=completed.stdout[: self.max_output_chars],
            stderr=completed.stderr[: self.max_output_chars],
            returncode=completed.returncode,
        )


class SourceRuntimeAdapter:
    """Run only the fenced Python blocks recognized by the upstream evaluator."""

    def __init__(self, sandbox: DeterministicPythonSandbox | None = None):
        self.sandbox = sandbox or DeterministicPythonSandbox()

    def extract_tool_calls(self, solver_text: str) -> tuple[PythonToolCall, ...]:
        reject_json_tool_call(solver_text)
        return tuple(PythonToolCall("PythonExec", block) for block in extract_python_blocks(solver_text))

    def execute_solver_text(
        self,
        solver_text: str,
    ) -> tuple[tuple[PythonToolCall, ...], list[dict[str, Any]], str | None]:
        calls = self.extract_tool_calls(solver_text)
        if not calls:
            return calls, [], None
        results = [self.sandbox.run(call.code.strip()).to_dict() for call in calls]
        return calls, results, format_code_execution_observation(results)

    def execute_as_message(self, solver_text: str) -> tuple[tuple[PythonToolCall, ...], list[dict[str, Any]], dict[str, str] | None]:
        calls, results, observation = self.execute_solver_text(solver_text)
        return calls, results, observation_message(results) if observation is not None else None


class ScriptedBackend:
    """Scripted Teacher backend with observable calls and failure injection."""

    def __init__(self, responses: Mapping[str, Iterable[Any]]):
        self._responses = {key: list(values) for key, values in responses.items()}
        self.calls: list[dict[str, Any]] = []

    def generate(self, payload: Mapping[str, Any]) -> Any:
        phase = str(payload.get("phase", "unknown"))
        self.calls.append(dict(payload))
        values = self._responses.get(phase, [])
        if not values:
            raise RuntimeError(f"no scripted response for phase {phase!r}")
        response = values.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


# --------------------------------------------------------------------------
# Frozen upstream sandbox
# --------------------------------------------------------------------------


class UpstreamSandboxUnavailable(RuntimeError):
    """Raised when the frozen upstream sandbox cannot be imported."""


def _agent0_vl_root() -> Path:
    # .../Agent0-VL/tools/local_sft_builder/runtime.py
    return Path(__file__).resolve().parents[2]


def _load_upstream_single_sandbox() -> Callable[..., Any]:
    """Import the frozen sandbox, honouring ``SANDBOX_ENDPOINT`` like upstream.

    ``sandbox/__init__.py`` selects the HTTP backend when ``SANDBOX_ENDPOINT``
    is set and the local subprocess backend otherwise; the single-snippet entry
    point is mirrored here because it reports ``timeout`` explicitly, whereas
    ``parallel_sandbox`` collapses every failure into a boolean.
    """

    root = str(_agent0_vl_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        if os.getenv("SANDBOX_ENDPOINT"):
            from sandbox.local_sandbox import single_sandbox
        else:
            from sandbox.subprocess_sandbox import single_sandbox
    except ImportError as exc:  # pragma: no cover - depends on target env
        raise UpstreamSandboxUnavailable(
            f"cannot import the upstream sandbox from {root}: {exc}"
        ) from exc
    return single_sandbox


def _normalize_newlines(value: str) -> str:
    """Normalize captured output to LF.

    The upstream sandbox decodes raw pipe bytes, so on Windows every captured
    line ends in CRLF while the frozen rollout runtime (Linux) sees LF.  Without
    this, the same Solver code would produce different ``[Code Execution
    Result]`` observations on the two platforms.  On Linux this is a no-op.
    """

    if "\r" not in value:
        return value
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _sandbox_result_from_upstream(result: Any) -> SandboxResult:
    if not isinstance(result, Mapping):
        raise RuntimeError("upstream sandbox returned a non-mapping result")
    run_result = result.get("run_result")
    if not isinstance(run_result, Mapping):
        run_result = {}
    status = result.get("status")
    if status not in {"success", "timeout", "runtime_error"}:
        status = "runtime_error"
    return SandboxResult(
        status=str(status),
        stdout=_normalize_newlines(str(run_result.get("stdout") or "")),
        stderr=_normalize_newlines(str(run_result.get("stderr") or "")),
    )


class UpstreamSandboxRunner:
    """Execute one Python snippet through the frozen upstream sandbox.

    Uses the same backend selection and the same result vocabulary as the
    rollout runtime, so ``[Code Execution Result]`` observations stay
    source-compatible.
    """

    backend_name = "upstream_agent0_vl_sandbox"

    def __init__(self, *, timeout_seconds: float | None = None):
        self.timeout_seconds = timeout_seconds
        self._single_sandbox = _load_upstream_single_sandbox()

    def run(self, code: str) -> SandboxResult:
        if not isinstance(code, str):
            raise TypeError("sandbox code must be text")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:  # pragma: no cover - the builder is synchronous
            raise RuntimeError(
                "UpstreamSandboxRunner is synchronous and cannot run inside an "
                "active event loop"
            )
        if self.timeout_seconds is None:
            result = asyncio.run(self._single_sandbox(code))
        else:
            result = asyncio.run(
                self._single_sandbox(code, run_timeout=self.timeout_seconds)
            )
        return _sandbox_result_from_upstream(result)

    def describe(self) -> dict[str, Any]:
        return {
            "sandbox_backend": self.backend_name,
            "sandbox_timeout_seconds": self.timeout_seconds,
            "sandbox_endpoint_configured": bool(os.getenv("SANDBOX_ENDPOINT")),
        }


class RealSandboxRuntime(SourceRuntimeAdapter):
    """``SourceRuntimeAdapter`` wired to the frozen upstream sandbox."""

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        sandbox: Any | None = None,
    ):
        super().__init__(
            sandbox=(
                sandbox
                if sandbox is not None
                else UpstreamSandboxRunner(timeout_seconds=timeout_seconds)
            )
        )
