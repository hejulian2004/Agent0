"""OpenAI SDK runtime for Responses-only remote inference."""

from __future__ import annotations

import base64
import copy
import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from openai import OpenAI

from .adapters import ResponsesAdapter
from .schema import CanonicalTrajectory, ProtocolError
from .tools import (
    ToolExecutionContext,
    ToolRegistry,
    execute_call_batch,
    get_tool_registry,
    input_image_from_items,
)


class CapabilityError(RuntimeError):
    """The configured endpoint cannot run this project's required protocol."""


@dataclass(frozen=True)
class ResponsesConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 180.0
    max_retries: int = 3
    max_tool_rounds: int = 8
    max_output_tokens: int = 2048

    @classmethod
    def from_env(cls) -> "ResponsesConfig":
        return cls(
            base_url=os.environ.get("AGENT0_RESPONSES_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("AGENT0_RESPONSES_API_KEY", ""),
            model=os.environ.get("AGENT0_RESPONSES_MODEL", ""),
            timeout_seconds=float(os.environ.get("AGENT0_RESPONSES_TIMEOUT_SECONDS", "180")),
            max_retries=int(os.environ.get("AGENT0_RESPONSES_MAX_RETRIES", "3")),
            max_tool_rounds=int(os.environ.get("AGENT0_RESPONSES_MAX_TOOL_ROUNDS", "8")),
            max_output_tokens=int(os.environ.get("AGENT0_RESPONSES_MAX_OUTPUT_TOKENS", "2048")),
        )

    def validate(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path.rstrip("/") != "/v1":
            raise ValueError("AGENT0_RESPONSES_BASE_URL must be an HTTP(S) /v1 API root")
        if not self.api_key or not self.model:
            raise ValueError("AGENT0_RESPONSES_API_KEY and AGENT0_RESPONSES_MODEL are required")
        if self.timeout_seconds <= 0 or self.max_retries < 0 or self.max_tool_rounds <= 0 or self.max_output_tokens <= 0:
            raise ValueError("invalid Responses timeout, retry, or tool-round limit")


class ResponsesRuntime:
    """Probe required capabilities, then run the canonical function loop."""

    def __init__(
        self,
        config: ResponsesConfig,
        registry: ToolRegistry | None = None,
        *,
        client: Any | None = None,
        probe_on_init: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        self.registry = registry or get_tool_registry()
        self.client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self.adapter = ResponsesAdapter()
        if probe_on_init:
            self.probe_capabilities()

    def _create(self, **kwargs: Any) -> Any:
        return self.client.responses.create(model=self.config.model, **kwargs)

    def probe_capabilities(self) -> None:
        """Require text, image, function calls and two sequential tool rounds."""
        try:
            text_response = self._create(input="Reply with OK.", max_output_tokens=64)
            if not any(item["type"] == "message" for item in self.adapter.output_items(text_response)):
                raise CapabilityError("text Responses output is missing")

            from PIL import Image

            image_buffer = io.BytesIO()
            Image.new("RGB", (1, 1), "red").save(image_buffer, format="PNG")
            image_url = "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")
            image_response = self._create(
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": "Describe this image briefly."},
                    {"type": "input_image", "image_url": image_url},
                ]}],
                max_output_tokens=64,
            )
            if not any(item["type"] == "message" for item in self.adapter.output_items(image_response)):
                raise CapabilityError("image Responses output is missing")

            probe_tool = {
                "type": "function",
                "name": "agent0_capability_probe",
                "description": "Return a small integer for protocol capability validation.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
            history: list[Any] = [{"role": "user", "content": "Call the capability probe in two consecutive rounds before giving your final answer."}]
            seen_ids: set[str] = set()
            for round_number in (1, 2):
                response = self._create(
                    input=history,
                    tools=[probe_tool],
                    tool_choice={"type": "function", "name": probe_tool["name"]},
                    max_output_tokens=128,
                )
                calls = [item for item in self.adapter.output_items(response) if item["type"] == "function_call"]
                if not calls:
                    raise CapabilityError(f"function_call missing in probe round {round_number}")
                history.extend(response.output)
                for call in calls:
                    if call["name"] != probe_tool["name"] or call["call_id"] in seen_ids:
                        raise CapabilityError("invalid or reused function call_id")
                    seen_ids.add(call["call_id"])
                    history.append(self.adapter.function_result({
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": {"success": True, "value": round_number},
                    }))
            final = self._create(input=history, tools=[probe_tool], tool_choice="none", max_output_tokens=64)
            if not any(item["type"] == "message" for item in self.adapter.output_items(final)):
                raise CapabilityError("endpoint did not continue after function_call_output")
        except CapabilityError:
            raise
        except Exception as exc:
            raise CapabilityError(f"Responses capability probe failed: {type(exc).__name__}: {exc}") from exc

    def run(
        self,
        initial_items: Sequence[Mapping[str, Any]],
        *,
        trajectory_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tool_context: Mapping[str, Any] | None = None,
    ) -> CanonicalTrajectory:
        trajectory = CanonicalTrajectory(
            trajectory_id=trajectory_id or f"traj_{uuid.uuid4().hex}",
            tools=self.registry.definitions(),
            metadata=copy.deepcopy(dict(metadata or {})),
        )
        for item in initial_items:
            trajectory.append(item)
        history: list[Any] = self.adapter.initial_input(initial_items)
        execution_context = ToolExecutionContext(tool_context)
        if not execution_context.get("current_image_path"):
            image = input_image_from_items(initial_items)
            if image is not None:
                try:
                    execution_context.set_current_image(image)
                except (TypeError, ValueError) as exc:
                    execution_context["current_image_error"] = str(exc)

        try:
            for _ in range(self.config.max_tool_rounds + 1):
                response = self._create(input=history, tools=trajectory.tools, max_output_tokens=self.config.max_output_tokens)
                output_items = self.adapter.output_items(response)
                calls = [item for item in output_items if item["type"] == "function_call"]
                for item in output_items:
                    trajectory.append(item)
                history.extend(response.output)
                if not calls:
                    if not any(item["type"] == "message" for item in output_items):
                        raise ProtocolError("Responses returned neither a message nor a function call")
                    trajectory.metadata["response_id"] = getattr(response, "id", None)
                    trajectory.validate()
                    return trajectory
                prev_img = execution_context.get("current_image_path")
                batch_results = execute_call_batch(self.registry, calls, execution_context)
                curr_img = execution_context.get("current_image_path")
                image_changed = (curr_img != prev_img) and (curr_img is not None)

                new_image_url: str | None = None
                if image_changed and Path(curr_img).is_file():
                    try:
                        raw_bytes = Path(curr_img).read_bytes()
                        new_image_url = f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('ascii')}"
                    except Exception:
                        new_image_url = None

                for idx, result in enumerate(batch_results):
                    img_to_attach = new_image_url if (idx == len(batch_results) - 1 and new_image_url) else None
                    trajectory.append(result)
                    history.append(self.adapter.function_result(result, image_url=img_to_attach))
            raise ProtocolError("Responses exceeded max_tool_rounds before a final message")
        finally:
            execution_context.close()

    def run_verifier(self, trajectory: CanonicalTrajectory) -> dict[str, Any]:
        """Run remote verifier role over a canonical trajectory via Responses API."""
        from verl.prompts.agent0_templates import render_verifier_request
        from .verifier import parse_verification_output

        prompt = render_verifier_request(trajectory)
        response = self._create(
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.config.max_output_tokens,
        )
        output_items = self.adapter.output_items(response)
        text = ""
        for item in output_items:
            if item.get("type") == "message":
                content = item.get("content", "")
                if isinstance(content, str):
                    text += content
                elif isinstance(content, list):
                    text += "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            elif item.get("type") == "reasoning":
                text += "".join(str(p.get("text", "")) for p in item.get("summary", []))
        return parse_verification_output(text) or {}

    def run_repair(
        self,
        trajectory: CanonicalTrajectory,
        feedback: Mapping[str, Any],
        *,
        tool_context: Mapping[str, Any] | None = None,
    ) -> CanonicalTrajectory:
        """Run remote repair role via Responses API applying local patch/continuation."""
        from verl.prompts.agent0_templates import render_repair_request

        prompt = render_repair_request(trajectory, feedback)
        return self.run(
            [{"type": "message", "role": "user", "content": prompt}],
            trajectory_id=f"{trajectory.trajectory_id}_repaired",
            metadata={"source_trajectory_id": trajectory.trajectory_id, "role": "repair", "feedback": dict(feedback)},
            tool_context=tool_context,
        )
