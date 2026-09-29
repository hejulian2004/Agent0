"""Factory for the sole Responses Teacher backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import TeacherBackend, TeacherConfig
from .openai_backend import ResponsesBackend


def create_teacher_backend(
    config: TeacherConfig | Mapping[str, Any] | None = None,
) -> TeacherBackend:
    """Create the configured backend.

    With no argument, configuration is read from ``AGENT0_RESPONSES_*``
    environment variables.  A mapping is accepted for tests and programmatic
    callers, while the API key remains an in-memory value only.
    """

    if config is None:
        resolved = TeacherConfig.from_env()
    elif isinstance(config, TeacherConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = TeacherConfig(**dict(config))
    else:
        raise TypeError("config must be TeacherConfig, mapping, or None")

    backend = resolved.normalized_backend
    if backend == "responses":
        return ResponsesBackend(resolved)
    raise ValueError(f"Unsupported Teacher backend {resolved.backend!r}; use responses")
