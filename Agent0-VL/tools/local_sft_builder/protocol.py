"""Adapter for the upstream Agent0-VL runtime protocol.

The upstream evaluator is the compatibility authority for Solver execution:
Solver emits fenced Python blocks, while Verifier and Repair emit JSON.  This
module does not introduce a JSON tool-call format of its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


PYTHON_BLOCK_RE = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n?(.*?)```", re.DOTALL
)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CONFIDENCE_RE = re.compile(
    r"CONFIDENCE:\s*([01](?:\.\d+)?|\.\d+)", re.IGNORECASE
)
FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER:\s*(.*)", re.IGNORECASE | re.DOTALL)


class ProtocolError(ValueError):
    """Raised when a response cannot be consumed by the source-compatible adapter."""


@dataclass(frozen=True)
class PythonToolCall:
    tool_name: str
    code: str


@dataclass(frozen=True)
class SolverResponse:
    text: str
    tool_calls: tuple[PythonToolCall, ...]
    confidence: float | None
    final_answer: str | None
    is_complete: bool


def extract_python_blocks(text: str) -> tuple[str, ...]:
    """Mirror the upstream evaluator's fenced-block extraction."""

    if not isinstance(text, str):
        raise ProtocolError("Solver response must be text")
    blocks = tuple(match.group(1) for match in PYTHON_BLOCK_RE.finditer(text))
    return tuple(block for block in blocks if block.strip())


def parse_solver_response(text: str) -> SolverResponse:
    blocks = extract_python_blocks(text)
    confidence_match = CONFIDENCE_RE.search(text)
    final_match = FINAL_ANSWER_RE.search(text)
    confidence = float(confidence_match.group(1)) if confidence_match else None
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ProtocolError("confidence must be in [0, 1]")
    final_answer = final_match.group(1).strip() if final_match else None
    return SolverResponse(
        text=text,
        tool_calls=tuple(PythonToolCall("PythonExec", block) for block in blocks),
        confidence=confidence,
        final_answer=final_answer,
        is_complete=bool(THINK_RE.search(text) and confidence is not None and final_answer),
    )


def validate_solver_final(text: str, *, max_reasoning_steps: int = 8) -> SolverResponse:
    response = parse_solver_response(text)
    if not response.is_complete:
        raise ProtocolError(
            "Solver final response must contain <think>, CONFIDENCE and FINAL_ANSWER"
        )
    if len(THINK_RE.findall(text)) > max_reasoning_steps:
        raise ProtocolError("Solver response exceeds max reasoning steps")
    return response


def _single_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ProtocolError("JSON response must be text")
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON response: {exc}") from exc
    if stripped[end:].strip():
        raise ProtocolError("response contains more than one JSON value")
    if not isinstance(value, dict):
        raise ProtocolError("response JSON must be an object")
    return value


def parse_verifier_json(text: str) -> dict[str, Any]:
    value = _single_json_object(text)
    required = {"step_index", "score", "confidence", "critique"}
    missing = required.difference(value)
    if missing:
        raise ProtocolError(f"verifier JSON missing fields: {sorted(missing)}")
    confidence = value["confidence"]
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ProtocolError("verifier confidence must be in [0, 1]")
    return value


def parse_repair_json(text: str) -> dict[str, Any]:
    value = _single_json_object(text)
    required = {"action", "target_step", "patch_type"}
    missing = required.difference(value)
    if missing:
        raise ProtocolError(f"repair JSON missing fields: {sorted(missing)}")
    if value["action"] not in {"PATCH", "NO_CHANGE"}:
        raise ProtocolError("repair action must be PATCH or NO_CHANGE")
    return value


def format_code_execution_observation(results: list[Mapping[str, Any]]) -> str:
    """Match the source evaluator's actual observation wrapper."""

    observation = "\n[Code Execution Result]\n"
    for result in results:
        run_result = result.get("run_result", {})
        stderr = str(run_result.get("stderr", ""))
        stdout = str(run_result.get("stdout", ""))
        if stderr:
            observation += f"Error: {stderr[-512:]}\n"
        elif stdout:
            observation += f"Output: {stdout[:512]}\n"
        else:
            observation += "No output\n"
    return observation


def observation_message(results: list[Mapping[str, Any]]) -> dict[str, str]:
    return {"role": "user", "content": format_code_execution_observation(results)}


def reject_json_tool_call(text: str) -> None:
    """Fail closed if a Solver emits the non-source JSON tool-call shape."""

    try:
        value = json.loads(text.strip())
    except (TypeError, json.JSONDecodeError):
        return
    if isinstance(value, dict) and {"tool_name", "tool_input"}.issubset(value):
        raise ProtocolError("Solver JSON tool calls are not part of the source runtime protocol")
