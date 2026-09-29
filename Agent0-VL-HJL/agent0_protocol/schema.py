"""One semantic trajectory format, independent of model tokens and HTTP JSON."""

from __future__ import annotations

import copy
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from jsonschema import Draft202012Validator


SCHEMA_VERSION = "agent0.responses.v1"
ITEM_TYPES = frozenset({"message", "reasoning", "function_call", "function_call_output"})


class ProtocolError(ValueError):
    """A semantic trajectory violates the Responses-style contract."""


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


@dataclass
class RawRollout:
    """Actual vLLM tokens and sampling values; never reconstructed from text."""

    prompt_token_ids: list[int]
    expanded_prompt_token_ids: list[int]
    response_token_ids: list[int]
    old_logprobs: list[float | None]
    response_mask: list[bool]
    attention_mask: list[bool]
    sampling_mask: list[bool]
    sampled_token_ids: list[int] = field(default_factory=list)
    sampling_metadata: dict[str, Any] = field(default_factory=dict)
    policy_version: str = ""
    model_version: str = ""

    def validate(self) -> None:
        length = len(self.response_token_ids)
        for name in ("old_logprobs", "response_mask", "sampling_mask"):
            if len(getattr(self, name)) != length:
                raise ProtocolError(f"rollout {name} is not aligned with response_token_ids")
        if len(self.attention_mask) != len(self.expanded_prompt_token_ids) + length:
            raise ProtocolError("rollout attention_mask is not aligned with token ids")
        for sampled, value in zip(self.sampling_mask, self.old_logprobs):
            if sampled and value is None:
                raise ProtocolError("sampled token is missing old_logprobs")
        if self.sampled_token_ids != [token for token, sampled in zip(self.response_token_ids, self.sampling_mask) if sampled]:
            raise ProtocolError("sampled_token_ids disagree with vLLM response tokens")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class CanonicalTrajectory:
    """The model-visible tool snapshot and ordered semantic items."""

    trajectory_id: str
    tools: list[dict[str, Any]]
    items: list[dict[str, Any]] = field(default_factory=list)
    rollout: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def append(self, item: Mapping[str, Any]) -> None:
        value = copy.deepcopy(dict(item))
        if value.get("type") not in ITEM_TYPES:
            raise ProtocolError(f"unknown item type: {value.get('type')!r}")
        self.items.append(value)

    def validate(self, *, complete: bool = True) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ProtocolError(f"unsupported schema_version: {self.schema_version!r}")
        if not self.trajectory_id:
            raise ProtocolError("trajectory_id is required")
        specs: dict[str, dict[str, Any]] = {}
        for tool in self.tools:
            if tool.get("type") != "function" or not tool.get("name"):
                raise ProtocolError("tools must be named function definitions")
            name = str(tool["name"])
            if name in specs:
                raise ProtocolError(f"duplicate tool definition: {name}")
            if not isinstance(tool.get("description"), str):
                raise ProtocolError(f"tool {name} needs a description")
            if tool.get("strict") is not True:
                raise ProtocolError(f"tool {name} must declare strict=true")
            parameters = tool.get("parameters")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ProtocolError(f"tool {name} parameters must be a JSON object schema")
            Draft202012Validator.check_schema(parameters)
            specs[name] = tool

        pending: dict[str, str] = {}
        used: set[str] = set()
        for item in self.items:
            kind = item.get("type")
            if kind not in ITEM_TYPES:
                raise ProtocolError(f"unknown item type: {kind!r}")
            if kind == "message":
                if item.get("role") not in {"system", "developer", "user", "assistant"}:
                    raise ProtocolError("message role must be system/developer/user/assistant")
                if not isinstance(item.get("content"), (str, list)):
                    raise ProtocolError("message content must be text or content parts")
            elif kind == "reasoning":
                if not isinstance(item.get("summary"), list):
                    raise ProtocolError("reasoning summary must be a list")
            elif kind == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                arguments = item.get("arguments")
                if not isinstance(call_id, str) or not call_id or call_id in used:
                    raise ProtocolError("function_call needs a unique call_id")
                if name not in specs:
                    raise ProtocolError(f"unregistered function: {name!r}")
                if not isinstance(arguments, dict):
                    raise ProtocolError("function_call arguments must be an object")
                Draft202012Validator(specs[name]["parameters"]).validate(arguments)
                pending[call_id] = str(name)
                used.add(call_id)
            else:
                call_id = item.get("call_id")
                if call_id not in pending:
                    raise ProtocolError(f"orphan or duplicate function_call_output: {call_id!r}")
                if not isinstance(item.get("output"), dict):
                    raise ProtocolError("function_call_output output must be an object")
                del pending[call_id]
        if complete and pending:
            raise ProtocolError(f"missing function_call_output for: {sorted(pending)}")
        if self.rollout:
            RawRollout(**self.rollout).validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalTrajectory":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ProtocolError("only agent0.responses.v1 trajectories are accepted")
        trajectory = cls(
            trajectory_id=str(value["trajectory_id"]),
            tools=copy.deepcopy(value["tools"]),
            items=copy.deepcopy(value["items"]),
            rollout=copy.deepcopy(value.get("rollout", {})),
            metadata=copy.deepcopy(value.get("metadata", {})),
        )
        trajectory.validate()
        return trajectory
