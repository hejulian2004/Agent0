"""OpenAI-compatible Teacher backend.

This module only talks to the model endpoint.  It deliberately owns no
trajectory, validation, projection, or retry policy:

* every real request must already hold a ``TeacherRequestBudget`` slot before
  ``generate`` is called, and a retry is a *new* slot;
* the backend never retries on its own;
* the credential is read from an environment variable by name and is never
  logged, returned, or written into a manifest.

The implementation uses the standard library so that no new runtime dependency
is required for the vcc environment.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_API_KEY_ENV = "AGENT0_TEACHER_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 300.0
_IMAGE_TOKEN = "<image>"
_MAX_ERROR_BODY_CHARS = 200


class TeacherBackendError(RuntimeError):
    """Raised when the Teacher endpoint fails in a non-timeout way."""


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters for one Teacher request."""

    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 1024
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    seed: int | None = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "request_timeout": self.timeout,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TeacherResponse:
    """One decoded Teacher response."""

    text: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    backend: str = "openai_compatible"
    # Logical correlation id for audit only.  The authoritative request id is
    # the one owned by the SQLite ledger lease.
    logical_request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "backend": self.backend,
            "logical_request_id": self.logical_request_id,
            "text_chars": len(self.text),
        }


def image_data_url(path: str | Path) -> str:
    """Encode a local image file as an ``image_url`` data URL."""

    image_path = Path(path)
    try:
        payload = image_path.read_bytes()
    except OSError as exc:
        raise TeacherBackendError(f"cannot read image for Teacher request: {image_path}") from exc
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_request_messages(
    messages: Sequence[Mapping[str, Any]],
    images: Sequence[str | Path],
) -> list[dict[str, Any]]:
    """Expand ``<image>`` placeholders into content blocks, in order.

    Images are consumed across the whole conversation in message order, so a
    multi-turn rollout keeps the original question's image attached.  The number
    of placeholders must equal the number of supplied images.
    """

    image_paths = list(images)
    cursor = 0
    built: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TeacherBackendError(
                "teacher messages must be objects with string role and content"
            )
        occurrences = content.count(_IMAGE_TOKEN)
        if occurrences == 0:
            built.append({"role": role, "content": content})
            continue
        blocks: list[dict[str, Any]] = []
        remaining = content
        for _ in range(occurrences):
            head, _, tail = remaining.partition(_IMAGE_TOKEN)
            if head:
                blocks.append({"type": "text", "text": head})
            if cursor >= len(image_paths):
                raise TeacherBackendError(
                    "conversation has more <image> placeholders than supplied images"
                )
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image_paths[cursor])},
                }
            )
            cursor += 1
            remaining = tail
        if remaining:
            blocks.append({"type": "text", "text": remaining})
        built.append({"role": role, "content": blocks})

    if cursor != len(image_paths):
        raise TeacherBackendError(
            f"{len(image_paths) - cursor} supplied image(s) were not referenced by any placeholder"
        )
    return built


class TeacherBackend:
    """One-attempt OpenAI-compatible chat-completions client."""

    backend_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        default_sampling: SamplingConfig | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError("api_key_env must be a non-empty string")
        self.base_url = base_url.strip()
        self.model = model.strip()
        self.api_key_env = api_key_env.strip()
        self.default_sampling = default_sampling or SamplingConfig()
        self.extra_body = dict(extra_body or {})

    @property
    def completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def describe(self) -> dict[str, Any]:
        """Non-secret description for logs and manifests.

        Field names avoid secret-like substrings so the manifest guard accepts
        them; the credential itself is never included.
        """

        return {
            "teacher_provider": self.backend_name,
            "teacher_model": self.model,
            "teacher_base_url": self.base_url,
            "teacher_completions_url": self.completions_url,
            "teacher_credential_env": self.api_key_env,
            "teacher_credential_configured": bool(os.environ.get(self.api_key_env)),
        }

    def generate(
        self,
        *,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        images: Sequence[str | Path] = (),
        request_id: str | None = None,
        sampling: SamplingConfig | None = None,
    ) -> TeacherResponse:
        """Issue exactly one Teacher request.

        Raises ``TimeoutError`` on timeout so the ledger records ``timeout``,
        and ``TeacherBackendError`` for every other failure so the ledger
        records ``other_failed``.  Never retries.
        """

        config = sampling or self.default_sampling
        request_messages = build_request_messages(messages, images)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "stream": False,
        }
        if config.seed is not None:
            payload["seed"] = config.seed
        payload.update(self.extra_body)

        response_data, response_headers = self._post_json(payload, timeout=config.timeout)

        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TeacherBackendError("teacher response has no choices")
        first = choices[0] if isinstance(choices[0], Mapping) else {}
        message = first.get("message") if isinstance(first, Mapping) else None
        if not isinstance(message, Mapping):
            raise TeacherBackendError("teacher response choice has no message object")
        text = message.get("content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise TeacherBackendError("teacher response content is not text")

        usage = response_data.get("usage")
        finish_reason = first.get("finish_reason") if isinstance(first, Mapping) else None
        return TeacherResponse(
            text=text,
            model=str(response_data.get("model") or self.model),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            backend=self.backend_name,
            logical_request_id=request_id
            or response_headers.get("x-request-id")
            or (str(response_data.get("id")) if response_data.get("id") else None),
        )

    def generate_from_payload(self, payload: Mapping[str, Any]) -> TeacherResponse:
        """Adapter matching ``TeacherRequestBudget.execute``'s callable contract."""

        sampling = payload.get("sampling")
        return self.generate(
            role=str(payload.get("role", "unknown")),
            messages=payload.get("messages") or (),
            images=payload.get("images") or (),
            request_id=payload.get("request_id"),
            sampling=sampling if isinstance(sampling, SamplingConfig) else None,
        )

    def _post_json(
        self,
        payload: Mapping[str, Any],
        *,
        timeout: float,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        credential = os.environ.get(self.api_key_env)
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        request = urllib.request.Request(
            self.completions_url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                response_headers = {key.lower(): value for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read()[:_MAX_ERROR_BODY_CHARS].decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - diagnostics must never mask the failure
                detail = ""
            raise TeacherBackendError(
                f"teacher endpoint returned HTTP {exc.code}: {detail}".strip()
            ) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError(
                    f"teacher request exceeded {timeout}s"
                ) from None
            raise TeacherBackendError(f"teacher endpoint unreachable: {exc.reason}") from None
        except (TimeoutError, socket.timeout):
            raise TimeoutError(f"teacher request exceeded {timeout}s") from None

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TeacherBackendError(f"teacher response is not valid JSON: {exc}") from None
        if not isinstance(decoded, dict):
            raise TeacherBackendError("teacher response is not a JSON object")
        return decoded, response_headers
