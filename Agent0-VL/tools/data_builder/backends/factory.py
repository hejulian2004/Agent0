"""Factory for the one-line Teacher backend switch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import TeacherBackend, TeacherConfig
from .hf_backend import LocalHFBackend
from .openai_backend import OpenAICompatibleBackend


def create_teacher_backend(
    config: TeacherConfig | Mapping[str, Any] | None = None,
) -> TeacherBackend:
    """Create the configured backend.

    With no argument, configuration is read from ``AGENT0_TEACHER_*``
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
    if backend == "openai_compatible":
        return OpenAICompatibleBackend(resolved)
    if backend == "hf":
        return LocalHFBackend(resolved)
    raise ValueError(
        f"Unsupported Teacher backend {resolved.backend!r}; "
        "use openai_compatible or hf"
    )
