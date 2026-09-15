"""Shared Teacher backend contracts and configuration.

The data builder can generate trajectories with either a remote
OpenAI-compatible endpoint or a local Hugging Face/VLM checkpoint.  This file
contains only the stable interface and small image/context conversion helpers;
provider-specific code lives in sibling modules.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from tools.data_builder.schema import ConversationState, ImageAsset


TeacherRole = Literal["solver", "verifier", "repair", "regeneration"]
TeacherBackendName = Literal["openai_compatible", "hf"]


class TeacherBackendError(RuntimeError):
    """Raised when a Teacher backend cannot produce a generation chunk."""


@dataclass
class TeacherConfig:
    """Configuration shared by all Teacher backends.

    ``api_key`` is intentionally excluded from ``repr`` and from
    :meth:`public_dict`.  It should be supplied through an environment
    variable, never written into a dataset, manifest, or source file.
    """

    backend: str = "openai_compatible"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    allow_anonymous: bool = False
    checkpoint: str | None = None
    revision: str | None = None
    timeout_seconds: float = 180.0
    max_retries: int = 3
    max_tokens: int = 2048
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = True
    device: str = "auto"
    device_map: str | None = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    seed: int = 42
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "TeacherConfig":
        """Load the quick-switch configuration from environment variables.

        Preferred names are ``AGENT0_TEACHER_*``.  ``OPENAI_API_KEY`` and
        ``OPENAI_BASE_URL`` are accepted as convenient aliases for the remote
        backend, but the task-specific names are recommended for experiments.
        """

        def env(name: str, *aliases: str) -> str | None:
            for key in (name, *aliases):
                value = os.getenv(key)
                if value is not None and value != "":
                    return value
            return None

        def as_bool(name: str, default: bool) -> bool:
            value = env(name)
            if value is None:
                return default
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{name} must be a boolean, got {value!r}")

        def as_int(name: str, default: int) -> int:
            value = env(name)
            return default if value is None else int(value)

        def as_float(name: str, default: float) -> float:
            value = env(name)
            return default if value is None else float(value)

        device_map = env("AGENT0_TEACHER_DEVICE_MAP")
        if device_map is None:
            device_map = "auto"
        elif device_map.lower() in {"none", "null", "off"}:
            device_map = None

        extra_body: dict[str, Any] = {}
        extra_body_text = env("AGENT0_TEACHER_EXTRA_BODY")
        if extra_body_text:
            parsed = json.loads(extra_body_text)
            if not isinstance(parsed, dict):
                raise ValueError("AGENT0_TEACHER_EXTRA_BODY must be a JSON object")
            extra_body = parsed

        return cls(
            backend=env("AGENT0_TEACHER_BACKEND") or "openai_compatible",
            model=env("AGENT0_TEACHER_MODEL"),
            base_url=env(
                "AGENT0_TEACHER_BASE_URL",
                "OPENAI_BASE_URL",
            ),
            api_key=env(
                "AGENT0_TEACHER_API_KEY",
                "OPENAI_API_KEY",
            ),
            allow_anonymous=as_bool("AGENT0_TEACHER_ALLOW_ANONYMOUS", False),
            checkpoint=env("AGENT0_TEACHER_CHECKPOINT"),
            revision=env("AGENT0_TEACHER_REVISION"),
            timeout_seconds=as_float("AGENT0_TEACHER_TIMEOUT_SECONDS", 180.0),
            max_retries=as_int("AGENT0_TEACHER_MAX_RETRIES", 3),
            max_tokens=as_int("AGENT0_TEACHER_MAX_TOKENS", 2048),
            temperature=as_float("AGENT0_TEACHER_TEMPERATURE", 0.2),
            top_p=as_float("AGENT0_TEACHER_TOP_P", 0.95),
            do_sample=as_bool("AGENT0_TEACHER_DO_SAMPLE", True),
            device=env("AGENT0_TEACHER_DEVICE") or "auto",
            device_map=device_map,
            dtype=env("AGENT0_TEACHER_DTYPE") or "auto",
            trust_remote_code=as_bool("AGENT0_TEACHER_TRUST_REMOTE_CODE", False),
            seed=as_int("AGENT0_TEACHER_SEED", 42),
            extra_body=extra_body,
        )

    def validate(self) -> None:
        backend = self.normalized_backend
        if backend == "openai_compatible":
            if not self.base_url:
                raise ValueError(
                    "OpenAI-compatible Teacher requires "
                    "AGENT0_TEACHER_BASE_URL"
                )
            if not self.model:
                raise ValueError(
                    "OpenAI-compatible Teacher requires AGENT0_TEACHER_MODEL"
                )
            if not self.api_key and not self.allow_anonymous:
                raise ValueError(
                    "OpenAI-compatible Teacher requires "
                    "AGENT0_TEACHER_API_KEY (or explicitly enable "
                    "AGENT0_TEACHER_ALLOW_ANONYMOUS=true)"
                )
        elif backend == "hf":
            if not self.checkpoint:
                raise ValueError(
                    "Local HF Teacher requires AGENT0_TEACHER_CHECKPOINT"
                )
        else:
            raise ValueError(
                "Unsupported Teacher backend "
                f"{self.backend!r}; use 'openai_compatible' or 'hf'"
            )

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def normalized_backend(self) -> str:
        aliases = {
            "openai": "openai_compatible",
            "api": "openai_compatible",
            "openai-compatible": "openai_compatible",
            "local": "hf",
            "local_hf": "hf",
            "huggingface": "hf",
        }
        return aliases.get(self.backend.strip().lower(), self.backend.strip().lower())

    def public_dict(self) -> dict[str, Any]:
        """Return auditable config metadata with secrets redacted."""

        return {
            "backend": self.normalized_backend,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "allow_anonymous": self.allow_anonymous,
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "device": self.device,
            "device_map": self.device_map,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "seed": self.seed,
            "extra_body": copy.deepcopy(self.extra_body),
        }


@dataclass
class GenerationChunk:
    """One response generated for one role at one generation boundary."""

    text: str
    role: TeacherRole
    backend: str
    model: str
    finish_reason: str | None = None
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "role": self.role,
            "backend": self.backend,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "request_id": self.request_id,
            "usage": copy.deepcopy(self.usage),
            "raw_response": self.raw_response,
            "metadata": copy.deepcopy(self.metadata),
        }


class TeacherBackend(ABC):
    """Backend interface consumed by the future tool-in-the-loop builder."""

    backend_name: str

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config

    @abstractmethod
    def generate_next(
        self,
        context: Sequence[Mapping[str, Any]] | ConversationState,
        role: TeacherRole,
        images: Sequence[Any] | None = None,
    ) -> GenerationChunk:
        """Generate exactly one current-step chunk."""

    def close(self) -> None:
        """Release backend resources; local models may override this."""

    def __enter__(self) -> "TeacherBackend":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def normalize_messages(
    context: Sequence[Mapping[str, Any]] | ConversationState,
) -> list[dict[str, Any]]:
    """Copy context messages without mutating ConversationState."""

    source = context.messages if isinstance(context, ConversationState) else context
    messages: list[dict[str, Any]] = []
    for message in source:
        if not isinstance(message, Mapping):
            raise TypeError("Teacher context messages must be mappings")
        role = str(message.get("role", "user"))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported chat role: {role!r}")
        copied = dict(message)
        copied["role"] = role
        copied.setdefault("content", "")
        messages.append(copied)
    return messages


def image_bytes_and_mime(image: Any) -> tuple[bytes, str]:
    """Convert supported ImageAsset/path/bytes/PIL values to bytes + MIME."""

    value = image
    if isinstance(value, ImageAsset):
        if value.bytes_data is not None:
            value = value.bytes_data
        elif value.path:
            value = Path(value.path).read_bytes()
        else:
            raise ValueError(f"ImageAsset {value.asset_id!r} has no bytes or path")
    elif isinstance(value, Mapping):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path"):
            value = Path(str(value["path"])).read_bytes()
        else:
            raise ValueError("Image mapping requires bytes or path")
    elif isinstance(value, (str, os.PathLike)):
        value = Path(value).read_bytes()

    if not isinstance(value, bytes):
        if isinstance(value, bytearray):
            value = bytes(value)
        elif hasattr(value, "save"):
            buffer = io.BytesIO()
            value.save(buffer, format="PNG")
            value = buffer.getvalue()
        else:
            raise TypeError(f"Unsupported image value: {type(value)!r}")

    if value.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif value.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif value.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif value.startswith(b"RIFF") and value[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    return value, mime


def image_data_url(image: Any) -> str:
    data, mime = image_bytes_and_mime(image)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def pil_images(images: Sequence[Any] | None) -> list[Any]:
    """Load image values as PIL objects for the local HF processor."""

    if images is None or len(images) == 0:
        return []
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - requirements include Pillow
        raise TeacherBackendError("Pillow is required for local VLM images") from exc

    result: list[Any] = []
    for image in images:
        data, _ = image_bytes_and_mime(image)
        with Image.open(io.BytesIO(data)) as opened:
            result.append(opened.convert("RGB").copy())
    return result
