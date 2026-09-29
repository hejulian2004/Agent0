"""Model-independent Agent0 prompt helpers for canonical trajectories."""

from .agent0_templates import (
    SOLVER_SYSTEM_PROMPT,
    assistant_text,
    render_repair_request,
    render_solver_request,
    render_system_prompt,
    render_verifier_request,
)

__all__ = [
    "SOLVER_SYSTEM_PROMPT",
    "assistant_text",
    "render_repair_request",
    "render_solver_request",
    "render_system_prompt",
    "render_verifier_request",
]
