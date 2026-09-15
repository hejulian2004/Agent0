# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The canonical Agent0-VL prompt protocol and parsers.

This module is the single source of truth for the Phase 0 protocol.  The
runtime, dataset builder, exporters, and generated ``scripts/prompt.txt``
must use these renderers/parsers rather than maintaining local variants.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.data_builder.schema import (
    MAX_OBSERVATION_TOKENS,
    MAX_TOOL_STDERR_CHARS,
    MAX_TOOL_STDOUT_CHARS,
    PROTOCOL_VERSION,
    SolverTurnType,
)


FINAL_ANSWER_PATTERN = re.compile(
    r"^[ \t]*FINAL_ANSWER:[ \t]*(?P<answer>[^\r\n]*)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
CONFIDENCE_PATTERN = re.compile(
    r"^[ \t]*CONFIDENCE:[ \t]*(?P<confidence>[-+]?(?:\d+(?:\.\d*)?|\.\d+))[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)\s*(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
CODE_START_PATTERN = re.compile(r"```(?:python|py)\b", re.IGNORECASE)


# ============================================================================
# Canonical system and role prompts
# ============================================================================

SOLVER_SYSTEM_PROMPT = f"""You are an advanced vision-language reasoning agent capable of multi-step reasoning and Python tool use.

Protocol version: {PROTOCOL_VERSION}

## Solver turn invariant

Each Solver generation is exactly one reasoning step. A Solver turn must be
one of these forms:

1. REASONING_TURN: reasoning only; no fenced Python and no FINAL_ANSWER.
2. TOOL_TURN: reasoning plus one or more fenced Python blocks; no FINAL_ANSWER.
3. FINAL_TURN: reasoning plus exactly one CONFIDENCE line and one FINAL_ANSWER
   line; no fenced Python.

Python and FINAL_ANSWER must never appear in the same Solver turn. A turn with
both is invalid and will not be executed.

## Python tool protocol

Only fenced Python is executable. Do not emit JSON tool calls. Use one or more
blocks in this form:

```python
print(...)
```

The sandbox output will be returned as:

[Code Execution Result]
Output: ...
Error: ...

If Python creates an image that must be used later, print:

PROCESSED_IMAGE: sandbox-relative/path.png

Do not treat tool output as a final answer. Only a later Solver FINAL_TURN
may finish the trajectory.

## Final answer protocol

When the problem is solved, use exactly:

CONFIDENCE: <number between 0 and 1>
FINAL_ANSWER: <single-line answer>

Do not use \\boxed{{...}} as the new final-answer format.

## Reasoning

Use <think>...</think> for reasoning when appropriate. Integrate actual
sandbox observations into later reasoning. Do not invent tool observations.
"""


VERIFIER_PROMPT_TEMPLATE = """You are a verification agent. Evaluate one Solver reasoning step using the actual task context and tool observation.

The system has already assigned a 1-based step index. Return exactly one JSON
object on one line with these fields:

{{
  "step_index": <integer from 1 to 8>,
  "score": <-1.0 to 1.0>,
  "confidence": <0.0 to 1.0>,
  "critique": "<at most 2 sentences>",
  "tool_check": <true or false>
}}

Check factual correctness, logical consistency, completeness, and whether any
tool output was used correctly. Do not decide whether the program should enter
Repair; that gate is controlled by the system.

## Solver step
{step_content}

## Actual tool observation
{tool_outputs}

Return JSON only.
"""


REPAIR_PROMPT_TEMPLATE = """You are a repair agent. The system has already determined that this step is eligible for repair because the verifier confidence is below the configured threshold.

Decide whether the specific critique warrants a local PATCH or NO_CHANGE. The
confidence threshold is controlled by the program; do not make a threshold
decision yourself.

Return exactly one JSON object on one line.

For a patch:
{{
  "action": "PATCH",
  "target_step": <1-based step number>,
  "patch_type": "text|code|tool_call|parameter",
  "new_content": "<minimal replacement content>",
  "justification": "<at most 2 sentences>"
}}

For no change:
{{
  "action": "NO_CHANGE",
  "target_step": <1-based step number>,
  "reason": "<why no patch is warranted>"
}}

## Verifier feedback
Critique: {critique}
Score: {score}
Confidence: {confidence}

## Original Solver step
{original_step}

Return JSON only.
"""


def render_system_prompt() -> str:
    return SOLVER_SYSTEM_PROMPT


def render_solver_request(question: str, image_context: str = "") -> str:
    parts = []
    if image_context:
        parts.append(f"## Available images\n{image_context}")
    parts.append(f"## Question\n{question}")
    parts.append("Begin the next Solver reasoning step.")
    return "\n\n".join(parts)


def truncate_chars(text: str | None, limit: int) -> str:
    """Apply the runtime-compatible character limit."""

    if not text:
        return ""
    return str(text)[:limit]


def render_observation_event(
    stdout: str = "",
    stderr: str = "",
    processed_image_count: int = 0,
) -> str:
    """Render a non-trainable observation using the canonical marker.

    Character truncation happens per field here.  The rollout is responsible
    for applying the second tokenizer-level max-observation limit.
    """

    stdout_text = truncate_chars(stdout, MAX_TOOL_STDOUT_CHARS)
    stderr_text = truncate_chars(stderr, MAX_TOOL_STDERR_CHARS)
    lines = ["[Code Execution Result]"]
    if stdout_text:
        lines.append(f"Output: {stdout_text}")
    if stderr_text:
        lines.append(f"Error: {stderr_text}")
    if not stdout_text and not stderr_text:
        lines.append("No output")
    if processed_image_count:
        lines.append("[Processed Image]")
        lines.extend(["<image>"] * int(processed_image_count))
    return "\n".join(lines)


def render_chat_messages(
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render protocol messages in a deterministic role-boundary format.

    The production rollout may pass this message list through the model's
    native chat template, but the message/event contract itself is fixed here
    and is used by the Phase 0 identity tests.
    """

    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        rendered.append("<|im_start|>assistant\n")
    return "".join(rendered)


def truncate_observation_tokens(
    text: str,
    tokenizer: Any,
    limit: int = MAX_OBSERVATION_TOKENS,
) -> list[int]:
    """Apply the second, tokenizer-level observation limit."""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return list(token_ids[:limit])


def render_verifier_request(step_content: str, tool_outputs: str = "None") -> str:
    return VERIFIER_PROMPT_TEMPLATE.format(
        step_content=step_content,
        tool_outputs=tool_outputs or "None",
    )


def render_repair_request(
    original_step: str,
    critique: str,
    score: float,
    confidence: float,
) -> str:
    return REPAIR_PROMPT_TEMPLATE.format(
        original_step=original_step,
        critique=critique,
        score=score,
        confidence=confidence,
    )


# Backward-compatible names used by the existing repository.
def get_solver_prompt(question: str, image_context: str = "") -> str:
    return render_system_prompt() + "\n\n" + render_solver_request(question, image_context)


get_verifier_prompt = render_verifier_request
get_repair_prompt = render_repair_request


# ============================================================================
# Canonical parsers
# ============================================================================


def _parse_float_lines(pattern: re.Pattern[str], text: str) -> list[float]:
    values: list[float] = []
    for match in pattern.finditer(text):
        try:
            values.append(float(match.group("confidence")))
        except (TypeError, ValueError):
            continue
    return values


def _parse_canonical_final_lines(text: str) -> list[str]:
    return [
        match.group("answer").strip()
        for match in FINAL_ANSWER_PATTERN.finditer(text)
    ]


def _parse_legacy_boxed(text: str) -> str | None:
    """Parse a balanced legacy ``\\boxed{...}`` expression."""

    marker = r"\boxed{"
    start = text.find(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = text[start + len(marker):index].strip()
                return answer or None
        index += 1
    return None


def parse_final_answer(
    text: str,
    *,
    allow_legacy_boxed: bool = False,
) -> str | None:
    """Parse canonical final answers, optionally supporting legacy boxed text."""

    canonical_answers = _parse_canonical_final_lines(text)
    if len(canonical_answers) == 1 and canonical_answers[0]:
        return canonical_answers[0]
    if len(canonical_answers) > 1:
        return None
    if allow_legacy_boxed:
        return _parse_legacy_boxed(text)
    return None


def parse_confidence(text: str) -> float | None:
    values = _parse_float_lines(CONFIDENCE_PATTERN, text)
    if len(values) != 1:
        return None
    value = values[0]
    if not 0.0 <= value <= 1.0:
        return None
    return value


def parse_code_blocks(text: str) -> list[str]:
    return [
        match.group("code").strip()
        for match in CODE_BLOCK_PATTERN.finditer(text)
    ]


@dataclass
class ParsedSolverTurn:
    """Structured result returned by :func:`parse_solver_turn`."""

    turn_type: SolverTurnType
    reasoning_text: str
    code_blocks: list[str]
    confidence: float | None
    final_answer: str | None
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return self.turn_type is not SolverTurnType.INVALID

def parse_solver_turn(
    text: str,
    *,
    allow_legacy_boxed: bool = False,
) -> ParsedSolverTurn:
    code_blocks = parse_code_blocks(text)
    final_lines = _parse_canonical_final_lines(text)
    confidence_values = _parse_float_lines(CONFIDENCE_PATTERN, text)
    errors: list[str] = []
    reasoning_text = text.strip()

    if CODE_START_PATTERN.search(text) and not code_blocks:
        errors.append("unclosed_or_unparseable_python_block")
        return ParsedSolverTurn(
            SolverTurnType.INVALID,
            reasoning_text,
            code_blocks,
            None,
            None,
            errors,
        )

    if len(final_lines) > 1:
        errors.append("multiple_final_answers")
        return ParsedSolverTurn(
            SolverTurnType.INVALID,
            reasoning_text,
            code_blocks,
            None,
            None,
            errors,
        )

    if final_lines:
        if not final_lines[0]:
            errors.append("final_answer_empty")
        if code_blocks:
            errors.append("tool_and_final_turn_are_mutually_exclusive")
        if len(confidence_values) != 1:
            errors.append("final_turn_requires_exactly_one_confidence")
        elif not 0.0 <= confidence_values[0] <= 1.0:
            errors.append("confidence_out_of_range")
        if errors:
            return ParsedSolverTurn(
                SolverTurnType.INVALID,
                reasoning_text,
                code_blocks,
                confidence_values[-1] if confidence_values else None,
                final_lines[0] or None,
                errors,
            )
        return ParsedSolverTurn(
            SolverTurnType.FINAL,
            reasoning_text,
            code_blocks,
            confidence_values[0],
            final_lines[0],
            errors,
        )

    if code_blocks:
        return ParsedSolverTurn(
            SolverTurnType.TOOL,
            reasoning_text,
            code_blocks,
            confidence_values[-1] if confidence_values else None,
            None,
            errors,
        )

    if allow_legacy_boxed:
        legacy_answer = parse_final_answer(text, allow_legacy_boxed=True)
        if legacy_answer is not None:
            return ParsedSolverTurn(
                SolverTurnType.FINAL,
                reasoning_text,
                [],
                confidence_values[-1] if confidence_values else None,
                legacy_answer,
                ["legacy_boxed_answer"],
            )
    elif _parse_legacy_boxed(text) is not None:
        errors.append("legacy_boxed_ignored")

    return ParsedSolverTurn(
        SolverTurnType.REASONING,
        reasoning_text,
        [],
        confidence_values[-1] if confidence_values else None,
        None,
        errors,
    )


def _iter_json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _last_json_object(text: str) -> dict[str, Any] | None:
    objects = list(_iter_json_objects(text))
    return objects[-1] if objects else None


def parse_verification(text: str) -> dict[str, Any] | None:
    data = _last_json_object(text)
    if data is None:
        return None
    required = {"step_index", "score", "confidence", "critique"}
    if not required.issubset(data):
        return None
    try:
        step_index = int(data["step_index"])
        score = float(data["score"])
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        return None
    if not 1 <= step_index <= 8 or not -1.0 <= score <= 1.0:
        return None
    if not 0.0 <= confidence <= 1.0 or not isinstance(data["critique"], str):
        return None
    tool_check = data.get("tool_check", False)
    if not isinstance(tool_check, bool):
        return None
    return {
        "step_index": step_index,
        "score": score,
        "confidence": confidence,
        "critique": data["critique"],
        "tool_check": tool_check,
    }


def parse_repair(text: str) -> dict[str, Any] | None:
    data = _last_json_object(text)
    if data is None or data.get("action") not in {"PATCH", "NO_CHANGE"}:
        return None
    try:
        target_step = int(data["target_step"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 1 <= target_step <= 8:
        return None

    action = data["action"]
    if action == "PATCH":
        if data.get("patch_type") not in {"text", "code", "tool_call", "parameter"}:
            return None
        if not isinstance(data.get("new_content"), str) or not data["new_content"].strip():
            return None
        if not isinstance(data.get("justification", ""), str):
            return None
        return {
            "action": "PATCH",
            "target_step": target_step,
            "patch_type": data["patch_type"],
            "new_content": data["new_content"],
            "justification": data.get("justification", ""),
        }

    if not isinstance(data.get("reason", ""), str):
        return None
    return {
        "action": "NO_CHANGE",
        "target_step": target_step,
        "reason": data.get("reason", ""),
    }


# Compatibility aliases for existing runtime imports.
parse_verification_output = parse_verification
parse_repair_instruction = parse_repair


def validate_solver_output(text: str) -> Dict[str, Any]:
    parsed = parse_solver_turn(text)
    return {
        "has_think_tags": bool(re.search(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE)),
        "has_confidence": bool(CONFIDENCE_PATTERN.search(text)),
        "has_final_answer": bool(_parse_canonical_final_lines(text)),
        "has_code": bool(parsed.code_blocks),
        "turn_type": parsed.turn_type.value,
        "errors": list(parsed.errors),
        "is_valid": parsed.is_valid,
    }


def validate_verifier_output(text: str) -> Dict[str, Any]:
    parsed = parse_verification(text)
    return {
        "has_json": parsed is not None,
        "has_all_fields": parsed is not None,
        "is_valid": parsed is not None,
    }


def validate_repair_output(text: str) -> Dict[str, Any]:
    parsed = parse_repair(text)
    return {
        "has_json": parsed is not None,
        "has_action": parsed is not None,
        "valid_action": parsed is not None,
        "is_valid": parsed is not None,
    }
