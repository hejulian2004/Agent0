"""Builder-owned Verifier, Repair, and regeneration prompts.

The Solver prompt is intentionally not copied here.  The official training
scripts pass ``scripts/prompt.txt`` as their system prompt, so the builder
reads that same file.  The two templates below mirror
``vllm_agent0_rollout_spmd.py::_load_prompt_templates`` from the pinned
release; they are the prompts actually injected by the released runtime.
"""

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


REGENERATE_PROMPT_TEMPLATE = """Now switch back to the Solver role. Apply the repair instruction to the previous reasoning step.

Repair instruction:
{repair_instruction}

Regenerate only the corrected local reasoning segment. Preserve validated context.
If computation or image inspection is needed, emit a fenced Python block, wait for
the [Code Execution Result] observation, and then continue. Do not claim that the
repair succeeded without using the returned evidence."""


CONTINUE_SOLVER_PROMPT_TEMPLATE = """Now switch back to the Solver role and continue the solution from the validated context above.
Do not repeat the Verifier JSON. If another computation or image inspection is needed,
use a fenced Python block and wait for the [Code Execution Result] observation."""


STAGE_SOLVER_CONTRACTS = {
    1: """You are generating a high-quality Stage 1 visual tool-use trajectory.
The user question may contain one or more <image> markers. If an image is present,
you MUST inspect it with at least one fenced Python code block before answering.
Inside the sandbox, `image_path` is an already-defined Python variable pointing to
the first input image: write `Image.open(image_path)`, never `Image.open("image_path")`.
Use PIL, OpenCV, matplotlib, pytesseract, or ordinary Python as appropriate. Only
inspect that one image; do not list directories, scan unrelated files, or run OCR
across many files. Avoid `plt.show()` and print concise objective evidence. Do not give the final answer in the same message
as a code block: wait for the [Code Execution Result] observation, then continue
the reasoning and answer. End the final Solver response with:
<answer>
\\boxed{...}
</answer>""",
    2: """You are generating a high-quality Stage 2 mathematical code-reasoning
trajectory. Use at least one fenced Python code block to calculate or independently
check the answer, and make the code print its result. Do not give the final answer
in the same message as a code block: wait for the [Code Execution Result] observation,
then explain the result. End the final Solver response with:
<answer>
\\boxed{...}
</answer>""",
}


def load_solver_system_prompt(repo_root: str | Path | None = None) -> str:
    """Read the pinned repository's authoritative ``scripts/prompt.txt``."""

    if repo_root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = Path(repo_root).expanduser().resolve()
    prompt_path = root / "scripts" / "prompt.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Solver system prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_stage_solver_prompt(repo_root: str | Path | None, stage: int) -> str:
    """Append a builder-only stage contract without changing ``prompt.txt``."""

    if stage not in STAGE_SOLVER_CONTRACTS:
        raise ValueError(f"Unsupported SFT stage: {stage}")
    base = load_solver_system_prompt(repo_root).rstrip()
    return f"{base}\n\n{STAGE_SOLVER_CONTRACTS[stage]}"


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
