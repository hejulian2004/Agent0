"""Teacher generation backends for Agent0-VL data construction."""

from .base import GenerationChunk, TeacherBackend, TeacherBackendError, TeacherConfig
from .factory import create_teacher_backend
from .hf_backend import LocalHFBackend
from .openai_backend import OpenAICompatibleBackend

__all__ = [
    "GenerationChunk",
    "LocalHFBackend",
    "OpenAICompatibleBackend",
    "TeacherBackend",
    "TeacherBackendError",
    "TeacherConfig",
    "create_teacher_backend",
]
