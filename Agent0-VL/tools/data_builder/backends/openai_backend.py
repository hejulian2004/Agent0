"""OpenAI-compatible Teacher backend.

The implementation uses the Python standard library instead of requiring a
provider SDK.  It works with OpenAI-compatible hosted APIs and with local
servers such as vLLM that expose ``/v1/chat/completions``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from .base import (
    GenerationChunk,
    TeacherBackend,
    TeacherBackendError,
    TeacherConfig,
    TeacherRole,
    image_data_url,
    normalize_messages,
)


class OpenAICompatibleBackend(TeacherBackend):
    """Generate one response from an OpenAI-compatible chat endpoint."""

    backend_name = "openai_compatible"

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__(config)
        if config.normalized_backend != self.backend_name:
            raise ValueError(
                "OpenAICompatibleBackend requires an openai_compatible config"
            )
        config.validate()

    @property
    def completions_url(self) -> str:
        assert self.config.base_url is not None
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def generate_next(
        self,
        context: Sequence[Mapping[str, Any]] | Any,
        role: TeacherRole,
        images: Sequence[Any] | None = None,
    ) -> GenerationChunk:
        messages = normalize_messages(context)
        image_values = list(images) if images is not None else []
        request_messages = self._with_images(messages, image_values)
        assert self.config.model is not None

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": request_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        payload.update(self.config.extra_body)

        response_data, response_headers = self._post_json(payload)
        text = self._extract_text(response_data)
        choices = response_data.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else {}
        if not isinstance(message, dict):
            message = {}

        request_id = response_headers.get("x-request-id") or response_data.get(
            "id"
        )
        model = str(response_data.get("model") or self.config.model)
        usage = response_data.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        return GenerationChunk(
            text=text,
            role=role,
            backend=self.backend_name,
            model=model,
            finish_reason=(
                str(first_choice.get("finish_reason"))
                if isinstance(first_choice, dict)
                and first_choice.get("finish_reason") is not None
                else None
            ),
            request_id=str(request_id) if request_id else None,
            usage=usage,
            raw_response=response_data,
            metadata={
                "backend": self.backend_name,
                "base_url": self.config.base_url,
                "requested_role": role,
                "message_count": len(request_messages),
                "image_count": len(image_values),
                "request_parameters": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens,
                    "stream": False,
                },
                "config": self.config.public_dict(),
            },
        )

    def _post_json(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "agent0vl-data-builder/1.0",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = urllib.request.Request(
            self.completions_url,
            data=body,
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    raw = response.read()
                    parsed = json.loads(raw.decode("utf-8"))
                    if not isinstance(parsed, dict):
                        raise TeacherBackendError("Teacher response must be a JSON object")
                    response_headers = {
                        str(key).lower(): str(value)
                        for key, value in response.headers.items()
                    }
                    return parsed, response_headers
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = TeacherBackendError(
                    f"Teacher HTTP {exc.code}: {error_body[-1000:]}"
                )
                if exc.code not in {408, 409, 429} and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < self.config.max_retries:
                time.sleep(min(2.0**attempt, 30.0))

        raise TeacherBackendError(
            f"OpenAI-compatible Teacher request failed after "
            f"{self.config.max_retries + 1} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _with_images(
        messages: list[dict[str, Any]],
        images: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """Attach current ConversationState images to the last user turn."""

        if not images:
            return messages
        user_index = next(
            (index for index in range(len(messages) - 1, -1, -1)
             if messages[index].get("role") == "user"),
            None,
        )
        if user_index is None:
            messages.append({"role": "user", "content": ""})
            user_index = len(messages) - 1

        message = messages[user_index]
        content = message.get("content", "")
        if isinstance(content, list):
            parts = list(content)
        else:
            parts = [{"type": "text", "text": str(content)}] if content else []
        parts.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(image)},
            }
            for image in images
        )
        message["content"] = parts
        return messages

    @staticmethod
    def _extract_text(response_data: Mapping[str, Any]) -> str:
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TeacherBackendError("Teacher response has no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise TeacherBackendError("Teacher response choice is not an object")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise TeacherBackendError("Teacher response has no message")
        content = message.get("content", "")
        if isinstance(content, str):
            if not content:
                raise TeacherBackendError("Teacher response content is empty")
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text":
                    value = part.get("text")
                    if value:
                        text_parts.append(str(value))
            text = "".join(text_parts)
            if text:
                return text
        raise TeacherBackendError("Teacher response has no text content")
