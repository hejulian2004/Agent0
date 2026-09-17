"""Small deterministic runtime used by the fake builder and its tests."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .protocol import (
    PythonToolCall,
    extract_python_blocks,
    format_code_execution_observation,
    observation_message,
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
