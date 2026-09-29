"""Shared Teacher backend contracts and configuration.

The data builder generates trajectories through a Responses endpoint.
This file contains the stable interface and image conversion helpers.
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
TeacherBackendName = Literal["responses"]


class TeacherBackendError(RuntimeError):
    """Raised when a Teacher backend cannot produce a generation chunk."""


@dataclass
class TeacherConfig:
    """Configuration shared by all Teacher backends.

    ``api_key`` is intentionally excluded from ``repr`` and from
    :meth:`public_dict`.  It should be supplied through an environment
    variable, never written into a dataset, manifest, or source file.
    """

    backend: str = "responses"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 180.0
    max_retries: int = 3
    max_tokens: int = 2048
    @classmethod
    def from_env(cls) -> "TeacherConfig":
        """Load the sole remote configuration namespace; no aliases are accepted."""
        return cls(
            backend="responses",
            model=os.getenv("AGENT0_RESPONSES_MODEL"),
            base_url=os.getenv("AGENT0_RESPONSES_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("AGENT0_RESPONSES_API_KEY"),
            timeout_seconds=float(os.getenv("AGENT0_RESPONSES_TIMEOUT_SECONDS", "180")),
            max_retries=int(os.getenv("AGENT0_RESPONSES_MAX_RETRIES", "3")),
            max_tokens=int(os.getenv("AGENT0_RESPONSES_MAX_OUTPUT_TOKENS", "2048")),
        )

    def validate(self) -> None:
        backend = self.normalized_backend
        if backend == "responses":
            if not self.base_url:
                raise ValueError("Responses Teacher requires AGENT0_RESPONSES_BASE_URL")
            if not self.model:
                raise ValueError("Responses Teacher requires AGENT0_RESPONSES_MODEL")
            if not self.api_key:
                raise ValueError("Responses Teacher requires AGENT0_RESPONSES_API_KEY")
        else:
            raise ValueError(f"Unsupported Teacher backend {self.backend!r}; use responses")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @property
    def normalized_backend(self) -> str:
        return self.backend.strip().lower()

    def public_dict(self) -> dict[str, Any]:
        """Return auditable config metadata with secrets redacted."""

        return {
            "backend": self.normalized_backend,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
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
    items: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
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
            "items": copy.deepcopy(self.items),
            "tools": copy.deepcopy(self.tools),
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
        if role not in {"system", "developer", "user", "assistant"}:
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
