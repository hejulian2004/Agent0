"""OpenAI SDK backend using the Responses API and canonical function items."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
from .base import (
    GenerationChunk,
    TeacherBackend,
    TeacherConfig,
    TeacherRole,
    image_data_url,
    normalize_messages,
)


class ResponsesBackend(TeacherBackend):
    backend_name = "responses"

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__(config)
        config.validate()
        if config.normalized_backend != self.backend_name:
            raise ValueError("ResponsesBackend requires backend=responses")
        self.runtime = ResponsesRuntime(
            ResponsesConfig(
                base_url=str(config.base_url),
                api_key=str(config.api_key),
                model=str(config.model),
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
                max_output_tokens=config.max_tokens,
            )
        )

    @staticmethod
    def _input_items(messages: list[dict[str, Any]], images: Sequence[Any]) -> list[dict[str, Any]]:
        if images:
            last_user = next((i for i in range(len(messages) - 1, -1, -1)
                              if messages[i]["role"] == "user"), None)
            if last_user is None:
                messages.append({"role": "user", "content": ""})
                last_user = len(messages) - 1
            original = messages[last_user]["content"]
            if isinstance(original, str):
                content = [{"type": "input_text", "text": original}]
            elif isinstance(original, list):
                content = [
                    {"type": "input_text", "text": str(part.get("text", ""))}
                    for part in original if isinstance(part, Mapping) and part.get("type") in {"text", "input_text"}
                ]
            else:
                raise TypeError("user content must be text or content parts")
            content.extend({"type": "input_image", "image_url": image_data_url(image)} for image in images)
            messages[last_user]["content"] = content
        return [{"type": "message", "role": message["role"], "content": message["content"]}
                for message in messages]

    def generate_next(
        self,
        context: Sequence[Mapping[str, Any]] | Any,
        role: TeacherRole,
        images: Sequence[Any] | None = None,
    ) -> GenerationChunk:
        messages = normalize_messages(context)
        initial_items = self._input_items(messages, images or [])
        trajectory = self.runtime.run(initial_items, metadata={"requested_role": role})
        final = next(
            item for item in reversed(trajectory.items)
            if item["type"] == "message" and item["role"] == "assistant"
        )
        content = final["content"]
        text = content if isinstance(content, str) else "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
        return GenerationChunk(
            text=text,
            role=role,
            backend=self.backend_name,
            model=str(self.config.model),
            finish_reason="completed",
            request_id=trajectory.metadata.get("response_id"),
            raw_response=trajectory.to_dict(),
            items=trajectory.items,
            tools=trajectory.tools,
            metadata={"image_count": len(images or []), "config": self.config.public_dict()},
        )

    def close(self) -> None:
        self.runtime.client.close()
