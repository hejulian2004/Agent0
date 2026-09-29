"""Responses-only Teacher configuration and image adapter checks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent0_protocol.schema import CanonicalTrajectory
from tools.data_builder.backends import ResponsesBackend, TeacherConfig, create_teacher_backend


def test_teacher_config_uses_only_responses_env(monkeypatch):
    monkeypatch.setenv("AGENT0_RESPONSES_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("AGENT0_RESPONSES_MODEL", "vision-model")
    monkeypatch.setenv("AGENT0_RESPONSES_API_KEY", "secret-key")
    config = TeacherConfig.from_env()
    config.validate()
    assert config.normalized_backend == "responses"
    assert "secret-key" not in repr(config)
    assert "secret-key" not in json.dumps(config.public_dict())


def test_old_backend_rejected():
    with pytest.raises(ValueError):
        TeacherConfig(backend="openai_compatible", model="m", base_url="https://example.test/v1", api_key="key").validate()


def test_image_part_uses_responses_shape():
    messages = [{"role": "user", "content": "What is shown?"}]
    items = ResponsesBackend._input_items(messages, [b"\x89PNG\r\n\x1a\nexample"])
    assert items[0]["content"][0] == {"type": "input_text", "text": "What is shown?"}
    assert items[0]["content"][1]["type"] == "input_image"
    assert items[0]["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_factory_requires_capability_probe(monkeypatch):
    def unavailable_probe(self):
        raise RuntimeError("endpoint has no Responses function support")
    monkeypatch.setattr("agent0_protocol.responses_runtime.ResponsesRuntime.probe_capabilities", unavailable_probe)
    with pytest.raises(RuntimeError, match="endpoint has no Responses"):
        create_teacher_backend(TeacherConfig(model="vision-model", base_url="https://example.test/v1", api_key="secret-key"))
