"""Wire and model-token adapters around the canonical trajectory."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .schema import ProtocolError, new_call_id


class ResponsesAdapter:
    """Convert only at the OpenAI Responses SDK boundary."""

    @staticmethod
    def output_items(response: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for value in response.output:
            raw = value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
            kind = raw.get("type")
            if kind == "function_call":
                raw_args = raw.get("arguments")
                if isinstance(raw_args, dict):
                    arguments = raw_args
                elif isinstance(raw_args, str):
                    try:
                        arguments = json.loads(raw_args)
                    except (KeyError, TypeError, json.JSONDecodeError) as exc:
                        raise ProtocolError("Responses function arguments are not valid JSON") from exc
                else:
                    raise ProtocolError("Responses function arguments must be a JSON object or valid JSON string")
                if not isinstance(arguments, dict):
                    raise ProtocolError("Responses function arguments must be a JSON object")
                items.append({
                    "type": "function_call",
                    "call_id": raw["call_id"],
                    "name": raw["name"],
                    "arguments": arguments,
                })
            elif kind == "message":
                content = []
                for part in raw.get("content", []):
                    if part.get("type") in {"output_text", "input_text"}:
                        content.append({"type": part["type"], "text": part.get("text", "")})
                    elif part.get("type") == "refusal":
                        content.append({"type": "refusal", "refusal": part.get("refusal", "")})
                items.append({"type": "message", "role": raw.get("role", "assistant"), "content": content})
            elif kind == "reasoning":
                summary = [
                    {"type": "summary_text", "text": str(part.get("text", ""))}
                    for part in raw.get("summary", [])
                    if isinstance(part, dict)
                ]
                items.append({"type": "reasoning", "summary": summary})
            else:
                raise ProtocolError(f"unsupported Responses output item: {kind!r}")
        return items

    @staticmethod
    def function_result(
        item: Mapping[str, Any],
        image_url: str | None = None,
    ) -> dict[str, Any]:
        if item.get("type") != "function_call_output" or not isinstance(item.get("output"), dict):
            raise ProtocolError("expected a structured function_call_output")
        output_dict = dict(item["output"])
        img_url = image_url or output_dict.get("image_url")
        if not img_url and output_dict.get("output_path"):
            try:
                from pathlib import Path
                import base64

                p = Path(output_dict["output_path"])
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    data = p.read_bytes()
                    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                    img_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            except Exception:
                pass

        if img_url:
            text_dict = {k: v for k, v in output_dict.items() if k not in {"image_url", "image_data"}}
            return {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": [
                    {"type": "input_text", "text": json.dumps(text_dict, ensure_ascii=False)},
                    {"type": "input_image", "image_url": img_url},
                ],
            }
        return {
            "type": "function_call_output",
            "call_id": item["call_id"],
            "output": json.dumps(output_dict, ensure_ascii=False),
        }

    @staticmethod
    def initial_input(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for item in items:
            if item.get("type") != "message" or item.get("role") not in {"system", "developer", "user"}:
                raise ProtocolError("initial Responses input must contain user/system/developer messages")
            result.append({"role": item["role"], "content": item["content"]})
        return result


class QwenModelAdapter:
    """Qwen chat template, tool-token parsing and tokenizer boundary.

    The caller passes genuine token IDs from vLLM. This adapter never encodes a
    decoded generation to reconstruct sampled IDs.
    """

    OPEN_CALL = "<tool_call>"
    CLOSE_CALL = "</tool_call>"
    OPEN_REASONING = "<think>"
    CLOSE_REASONING = "</think>"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def decode_items(self, generated_token_ids: list[int]) -> list[dict[str, Any]]:
        text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        for boundary in ("<|im_end|>", "<|endoftext|>"):
            text = text.replace(boundary, "")
        items: list[dict[str, Any]] = []
        while text:
            positions = [(text.find(marker), marker) for marker in (self.OPEN_REASONING, self.OPEN_CALL)]
            positions = [(position, marker) for position, marker in positions if position >= 0]
            if not positions:
                message = text.strip()
                if message:
                    items.append({"type": "message", "role": "assistant", "content": message})
                break
            position, marker = min(positions)
            before = text[:position].strip()
            if before:
                items.append({"type": "message", "role": "assistant", "content": before})
            closing = self.CLOSE_REASONING if marker == self.OPEN_REASONING else self.CLOSE_CALL
            payload, found, text = text[position + len(marker):].partition(closing)
            if not found:
                raise ProtocolError("unterminated Qwen structured model token")
            if marker == self.OPEN_REASONING:
                items.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": payload.strip()}]})
                continue
            try:
                call = json.loads(payload.strip())
            except json.JSONDecodeError as exc:
                raise ProtocolError("Qwen function-call token contains invalid JSON") from exc
            if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
                raise ProtocolError("Qwen function-call token needs name and object arguments")
            if not isinstance(call.get("name"), str):
                raise ProtocolError("Qwen function-call token needs a function name")
            items.append({
                "type": "function_call",
                "call_id": new_call_id(),
                "name": call["name"],
                "arguments": call["arguments"],
            })
        return items

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                "<image>" if part.get("type") == "input_image" else str(part.get("text", ""))
                for part in content if isinstance(part, dict)
            )
        raise ProtocolError("unsupported message content")

    def render(self, items: Iterable[Mapping[str, Any]], tools: list[dict[str, Any]], *, generate: bool) -> str:
        """Render semantic items using Qwen's model-specific message tokens."""
        parts: list[str] = []
        for item in items:
            kind = item.get("type")
            if kind == "message":
                role = item["role"]
                if role not in {"system", "developer", "user", "assistant"}:
                    raise ProtocolError("invalid semantic message role")
                content = self._content_text(item["content"])
                if role == "system" and tools:
                    if "Available functions:" not in content:
                        content += "\n\nAvailable functions:\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            elif kind == "reasoning":
                summary = "".join(str(part.get("text", "")) for part in item["summary"])
                parts.append(f"<|im_start|>assistant\n{self.OPEN_REASONING}{summary}{self.CLOSE_REASONING}<|im_end|>\n")
            elif kind == "function_call":
                payload = json.dumps({"name": item["name"], "arguments": item["arguments"]}, ensure_ascii=False, separators=(",", ":"))
                parts.append(f"<|im_start|>assistant\n{self.OPEN_CALL}{payload}{self.CLOSE_CALL}<|im_end|>\n")
            elif kind == "function_call_output":
                payload = json.dumps({"call_id": item["call_id"], "output": item["output"]}, ensure_ascii=False, separators=(",", ":"))
                parts.append(f"<|im_start|>tool\n{payload}<|im_end|>\n")
            else:
                raise ProtocolError(f"unsupported item type: {kind!r}")
        if generate:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode_context(self, items: Iterable[Mapping[str, Any]], tools: list[dict[str, Any]]) -> list[int]:
        return self.tokenizer.encode(self.render(items, tools, generate=True), add_special_tokens=False)

    def stop_token_ids(self) -> list[int]:
        tokens = ["<|im_end|>", "<|endoftext|>"]
        ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        return [value for value in ids if isinstance(value, int) and value >= 0]
