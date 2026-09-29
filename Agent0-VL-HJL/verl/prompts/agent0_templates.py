"""Model-independent Agent0 prompts; tool definitions come from the registry."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from agent0_protocol.schema import CanonicalTrajectory
from tools.data_builder.schema import PROTOCOL_VERSION


SOLVER_SYSTEM_PROMPT = f"""You are a vision-language reasoning agent.
Protocol: {PROTOCOL_VERSION}.
Use only the function tools supplied with the request. Ground each conclusion in
actual observations. Return a concise final assistant message after all tools
have completed. Function arguments are JSON objects, and every function result
is paired with its call_id by the runtime.
"""


def render_system_prompt() -> str:
    return SOLVER_SYSTEM_PROMPT


def render_solver_request(question: str, image_context: str = "") -> str:
    context = f"Image context: {image_context}\n" if image_context else ""
    return f"{context}Question:\n{question.strip()}"


def render_verifier_request(trajectory: CanonicalTrajectory) -> str:
    trajectory.validate()
    return (
        "Verify the following semantic trajectory. Check registered functions, "
        "arguments, call IDs, observations, and final answer. Return a JSON "
        "object with score, confidence, critique, and tool_check.\n"
        + json.dumps(trajectory.to_dict(), ensure_ascii=False)
    )


def render_repair_request(
    trajectory: CanonicalTrajectory,
    feedback: Mapping[str, Any],
) -> str:
    trajectory.validate()
    return (
        "Repair the specific issue using the registered functions. A repeated "
        "function invocation must receive a new call_id. Return the corrected "
        "assistant message or function call.\n"
        + json.dumps({"trajectory": trajectory.to_dict(), "feedback": dict(feedback)}, ensure_ascii=False)
    )


def assistant_text(items: Sequence[Mapping[str, Any]]) -> str | None:
    for item in reversed(items):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            value = "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
            return value.strip() or None
    return None
