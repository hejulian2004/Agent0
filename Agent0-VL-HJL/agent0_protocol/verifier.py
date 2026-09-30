"""Verifier and retry logic over semantic items, independent of model tokens."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import CanonicalTrajectory, ProtocolError, new_call_id
from .tools import ToolRegistry


@dataclass
class Verification:
    valid: bool
    issues: list[str] = field(default_factory=list)
    final_answer: str | None = None


def _text(item: dict[str, Any]) -> str:
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def verify_trajectory(
    trajectory: CanonicalTrajectory,
    registry: ToolRegistry,
    *,
    expected_answer: str | None = None,
) -> Verification:
    issues: list[str] = []
    try:
        trajectory.validate()
    except Exception as exc:
        issues.append(f"protocol: {exc}")
    current_definitions = {tool["name"]: tool for tool in registry.definitions()}
    for tool in trajectory.tools:
        if current_definitions.get(tool.get("name")) != tool:
            issues.append(f"tool definition does not match registry: {tool.get('name')}")
    calls = [item for item in trajectory.items if item.get("type") == "function_call"]
    outputs = [item for item in trajectory.items if item.get("type") == "function_call_output"]
    if len(calls) != len(outputs):
        issues.append("function calls and outputs have different counts")
    assistant_messages = [
        (index, _text(item))
        for index, item in enumerate(trajectory.items)
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    final_answer = assistant_messages[-1][1].strip() if assistant_messages else None
    if not final_answer:
        issues.append("final assistant message is missing")
    elif outputs and assistant_messages[-1][0] < max(
        index for index, item in enumerate(trajectory.items) if item.get("type") == "function_call_output"
    ):
        issues.append("final answer precedes the last tool observation")
    if expected_answer is not None and final_answer is not None:
        if final_answer.casefold().strip() != expected_answer.casefold().strip():
            issues.append("final answer disagrees with expected observation")
    for item in outputs:
        output = item.get("output", {})
        if isinstance(output, dict) and output.get("success") is False:
            issues.append(f"tool failed: {item.get('call_id')}")
    return Verification(valid=not issues, issues=issues, final_answer=final_answer)


def retry_function(
    trajectory: CanonicalTrajectory,
    original_call_id: str,
    registry: ToolRegistry,
    *,
    context: dict[str, Any] | None = None,
    rollback_checkpoint: Any | None = None,
) -> str:
    """Append a fresh semantic retry using a new call_id."""
    original = next(
        (item for item in trajectory.items
         if item.get("type") == "function_call" and item.get("call_id") == original_call_id),
        None,
    )
    if original is None:
        raise ProtocolError(f"cannot retry unknown call_id: {original_call_id}")
    if hasattr(context, "rollback") and rollback_checkpoint is not None:
        context.rollback(rollback_checkpoint)
    retry = copy.deepcopy(original)
    retry["call_id"] = new_call_id()
    trajectory.append(retry)
    result = registry.execute(retry, context)
    trajectory.append({"type": "function_call_output", "call_id": retry["call_id"], "output": result})
    trajectory.validate()
    return retry["call_id"]


def repair_function(
    trajectory: CanonicalTrajectory,
    original_call_id: str,
    registry: ToolRegistry,
    *,
    new_arguments: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    rollback_checkpoint: Any | None = None,
) -> str:
    """Append a repaired function call with optional argument changes and image rollback."""
    original = next(
        (item for item in trajectory.items
         if item.get("type") == "function_call" and item.get("call_id") == original_call_id),
        None,
    )
    if original is None:
        raise ProtocolError(f"cannot repair unknown call_id: {original_call_id}")
    if hasattr(context, "rollback") and rollback_checkpoint is not None:
        context.rollback(rollback_checkpoint)
    repair = copy.deepcopy(original)
    repair["call_id"] = new_call_id()
    if new_arguments is not None:
        repair["arguments"] = copy.deepcopy(new_arguments)
    trajectory.append(repair)
    result = registry.execute(repair, context)
    trajectory.append({"type": "function_call_output", "call_id": repair["call_id"], "output": result})
    trajectory.validate()
    return repair["call_id"]


def extract_json_dict(text: str) -> dict[str, Any] | None:
    """Robustly extract a JSON object from raw text, code fences, or mixed outputs."""
    cleaned = text.strip()
    try:
        val = json.loads(cleaned)
        if isinstance(val, dict):
            return val
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    for block in reversed(blocks):
        try:
            val = json.loads(block)
            if isinstance(val, dict):
                return val
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            val = json.loads(cleaned[start:end + 1])
            if isinstance(val, dict):
                return val
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None


def parse_verification_output(text: str) -> dict[str, Any] | None:
    """Parse structured verification evaluation (score, confidence, critique)."""
    val = extract_json_dict(text)
    if not isinstance(val, dict):
        return None
    step_key = "step_index" if "step_index" in val else ("step" if "step" in val else None)
    if "score" in val:
        try:
            score = max(-1.0, min(1.0, float(val["score"])))
            conf_val = val.get("confidence", 0.8)
            conf = max(0.0, min(1.0, float(conf_val)))
            res = dict(val)
            res["score"] = score
            res["confidence"] = conf
            if step_key and "step_index" not in res:
                res["step_index"] = val[step_key]
            return res
        except (ValueError, TypeError):
            pass
    return None


def parse_repair_instruction(text: str) -> dict[str, Any] | None:
    """Parse structured repair instruction (action: PATCH | NO_CHANGE)."""
    val = extract_json_dict(text)
    if isinstance(val, dict):
        action = str(val.get("action", "")).upper()
        if action in {"PATCH", "NO_CHANGE"}:
            res = dict(val)
            res["action"] = action
            return res
    return None
