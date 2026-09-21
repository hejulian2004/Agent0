"""Minimal OpenAI-compatible teacher backend used by the SFT builder.

The backend intentionally uses the standard library.  It speaks only the
``/v1/chat/completions`` shape needed by the fast reproduction and does not
add retries, provider abstractions, persistence, or scheduling.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


class TeacherError(RuntimeError):
    """Raised when the compatible chat endpoint cannot produce text."""


@dataclass(frozen=True)
class TeacherConfig:
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    timeout: float = 120.0
    retries: int = 2


def _image_data_url(image: str) -> str:
    """Return a URL accepted by OpenAI-compatible vision endpoints."""

    if image.startswith(("data:", "http://", "https://")):
        return image
    path = Path(image).expanduser()
    if not path.is_file():
        raise TeacherError(f"Image path does not exist: {image}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _text_parts(content: Any) -> List[Dict[str, str]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict)]
    return [{"type": "text", "text": str(content)}]


def _render_messages(messages: Sequence[Mapping[str, Any]], images: Sequence[str]) -> List[Dict[str, Any]]:
    """Convert internal ``<image>`` markers to vision message parts.

    The internal conversation remains plain strings for ms-swift export.  Only
    the request payload gets image parts, so source paths and image bytes are
    never inserted into the training messages themselves.
    """

    rendered: List[Dict[str, Any]] = []
    image_index = 0
    marker = re.compile(r"<image>", re.IGNORECASE)
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role != "user" or not isinstance(content, str) or "<image>" not in content.lower():
            rendered.append({"role": role, "content": content})
            continue

        parts: List[Dict[str, Any]] = []
        cursor = 0
        for match in marker.finditer(content):
            text = content[cursor:match.start()]
            if text:
                parts.append({"type": "text", "text": text})
            if image_index >= len(images):
                raise TeacherError("The conversation contains more <image> markers than images")
            parts.append({
                "type": "image_url",
                "image_url": {"url": _image_data_url(str(images[image_index]))},
            })
            image_index += 1
            cursor = match.end()
        tail = content[cursor:]
        if tail:
            parts.append({"type": "text", "text": tail})
        rendered.append({"role": role, "content": parts})

    if image_index != len(images):
        raise TeacherError(
            f"The conversation has {image_index} image markers but {len(images)} images"
        )
    return rendered


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    # Accept both an OpenAI base URL (already ending in /v1) and a bare
    # host:port URL used by a local vLLM server.
    if re.search(r"/v[0-9]+$", base):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _response_text(data: Mapping[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TeacherError(f"Teacher response has no choices[0].message.content: {data}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


class OpenAICompatibleTeacher:
    """One-request-at-a-time teacher for ``/v1/chat/completions``."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        retries: int = 2,
    ) -> None:
        self.config = TeacherConfig(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
        )

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        images: Sequence[str],
        system_prompt: str,
    ) -> str:
        """Generate one assistant text response without exposing source answers."""

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *_render_messages(messages, images),
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            _endpoint(self.config.base_url),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        api_key = os.getenv(self.config.api_key_env)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")

        hostname = urllib.parse.urlparse(request.full_url).hostname
        last_error: Optional[str] = None
        for attempt in range(self.config.retries + 1):
            try:
                if hostname in {"127.0.0.1", "localhost", "::1"}:
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    response_context = opener.open(request, timeout=self.config.timeout)
                else:
                    response_context = urllib.request.urlopen(request, timeout=self.config.timeout)
                with response_context as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if 500 <= exc.code < 600 and attempt < self.config.retries:
                    last_error = f"HTTP {exc.code}: {detail}"
                else:
                    raise TeacherError(f"Teacher HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                last_error = str(getattr(exc, "reason", exc))
                if attempt >= self.config.retries:
                    raise TeacherError(f"Teacher request failed after {attempt + 1} attempts: {last_error}") from exc
            if attempt < self.config.retries:
                time.sleep(min(2.0 ** attempt, 4.0))
        else:
            raise TeacherError(
                f"Teacher request failed after {self.config.retries + 1} attempts: {last_error or 'unknown error'}"
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TeacherError(f"Teacher returned invalid JSON: {raw[:2000]}") from exc
        return _response_text(data)


__all__ = ["OpenAICompatibleTeacher", "TeacherConfig", "TeacherError"]
