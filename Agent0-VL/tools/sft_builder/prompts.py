"""Read the released SFT system prompt and rollout role prompts."""

from __future__ import annotations

from pathlib import Path


VERIFIER_PROMPT_TEMPLATE = """Now switch to the Verifier role. Verify the reasoning step above using the available evidence.

Step to evaluate:
{step_content}

Tool outputs (if any):
{tool_outputs}

Principles:
- Ground verification on objective tool evidence.
- Penalize unsupported or inconsistent reasoning.
- High confidence requires agreement between tool and text.

Output exactly one JSON line:
{{"step_index": {step_index}, "score": <-1 to 1>, "confidence": <0 to 1>, "critique": "<at most 2 sentences>", "tool_check": <true|false>}}"""


REPAIR_PROMPT_TEMPLATE = """Now switch to the Self-Repair role. The Verifier flagged the reasoning step with low confidence.

Verification result:
- Score: {score}
- Confidence: {confidence}
- Critique: {critique}

Original step:
{original_step}

Propose a minimal local patch that fixes the specific error WITHOUT rewriting validated context.
Output exactly one JSON line, either:
{{"action": "PATCH", "target_step": {step_index}, "patch_type": "<text|code|tool_call|parameter>", "new_content": "<minimal replacement>", "justification": "<at most 2 sentences>"}}
or:
{{"action": "NO_CHANGE", "target_step": {step_index}, "reason": "<why repair is not warranted>"}}"""


REGENERATE_PROMPT_TEMPLATE = """A repair instruction has been issued for the previous step:
{repair_instruction}

Switch back to the Solver role. Re-derive the corrected reasoning step applying this patch, then continue toward the final answer."""


CONTINUE_SOLVER_PROMPT_TEMPLATE = "Continue solving the problem using the conversation so far."


def load_solver_system_prompt(repo_root: str | Path | None = None) -> str:
    """Read the released ``scripts/prompt.txt`` without adding instructions."""

    if repo_root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = Path(repo_root).expanduser().resolve()
    prompt_path = root / "scripts" / "prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Solver system prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_stage_solver_prompt(repo_root: str | Path | None, stage: int) -> str:
    """Use the same released system prompt for both SFT stages."""

    if stage not in (1, 2):
        raise ValueError(f"Unsupported SFT stage: {stage}")
    return load_solver_system_prompt(repo_root)


def verifier_prompt(step_content: str, tool_outputs: str | None, step_index: int) -> str:
    """Render the released runtime's Verifier injection."""

    return VERIFIER_PROMPT_TEMPLATE.format(
        step_content=step_content,
        tool_outputs=tool_outputs if tool_outputs else "None",
        step_index=step_index,
    )


def repair_prompt(
    original_step: str,
    critique: str,
    score: float,
    confidence: float,
    step_index: int,
) -> str:
    """Render the released runtime's confidence-gated Repair injection."""

    return REPAIR_PROMPT_TEMPLATE.format(
        original_step=original_step,
        critique=critique,
        score=score,
        confidence=confidence,
        step_index=step_index,
    )


def regenerate_prompt(repair_instruction: str) -> str:
    """Render the Solver prompt that follows a parsed PATCH instruction."""

    return REGENERATE_PROMPT_TEMPLATE.format(repair_instruction=repair_instruction)


def continue_solver_prompt() -> str:
    """Render an explicit Solver turn after a non-final post-repair check."""

    return CONTINUE_SOLVER_PROMPT_TEMPLATE


# Names matching the released template module make the small builder easy to
# use from notebooks without making that module part of the Solver path.
get_verifier_prompt = verifier_prompt
get_repair_prompt = repair_prompt
get_regenerate_prompt = regenerate_prompt
get_continue_solver_prompt = continue_solver_prompt


__all__ = [
    "REPAIR_PROMPT_TEMPLATE",
    "REGENERATE_PROMPT_TEMPLATE",
    "CONTINUE_SOLVER_PROMPT_TEMPLATE",
    "VERIFIER_PROMPT_TEMPLATE",
    "get_repair_prompt",
    "get_verifier_prompt",
    "get_regenerate_prompt",
    "get_continue_solver_prompt",
    "load_stage_solver_prompt",
    "load_solver_system_prompt",
    "repair_prompt",
    "regenerate_prompt",
    "continue_solver_prompt",
    "verifier_prompt",
]
