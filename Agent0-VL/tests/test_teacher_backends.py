"""Tests for the remote/local Teacher backend switch."""

from __future__ import annotations

import json

from tools.data_builder.backends import (
    LocalHFBackend,
    OpenAICompatibleBackend,
    TeacherConfig,
    create_teacher_backend,
)
from tools.data_builder.backends import openai_backend


def test_teacher_config_reads_remote_env_without_exposing_key(monkeypatch):
    monkeypatch.setenv("AGENT0_TEACHER_BACKEND", "openai_compatible")
    monkeypatch.setenv("AGENT0_TEACHER_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("AGENT0_TEACHER_MODEL", "vision-model")
    monkeypatch.setenv("AGENT0_TEACHER_API_KEY", "secret-key")

    config = TeacherConfig.from_env()
    config.validate()

    assert config.normalized_backend == "openai_compatible"
    assert config.public_dict()["api_key_configured"] is True
    assert "secret-key" not in repr(config)
    assert "secret-key" not in json.dumps(config.public_dict())


def test_factory_switches_to_local_hf_without_loading_model():
    backend = create_teacher_backend(
        {
            "backend": "local",
            "checkpoint": "./checkpoints/qwen2.5-vl",
        }
    )

    assert isinstance(backend, LocalHFBackend)
    assert backend.model is None


def test_factory_switches_to_openai_compatible():
    backend = create_teacher_backend(
        TeacherConfig(
            backend="openai",
            base_url="https://example.test/v1",
            model="vision-model",
            api_key="secret-key",
        )
    )

    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.completions_url == "https://example.test/v1/chat/completions"


def test_openai_backend_sends_key_and_vision_data(monkeypatch):
    captured: dict[str, object] = {}

    class FakeResponse:
        headers = {"x-request-id": "request-1"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "id": "chatcmpl-test",
                    "model": "vision-model",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "answer"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(openai_backend.urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleBackend(
        TeacherConfig(
            backend="openai_compatible",
            base_url="https://example.test/v1",
            model="vision-model",
            api_key="secret-key",
            timeout_seconds=12,
        )
    )
    png = b"\x89PNG\r\n\x1a\n" + b"minimal-png-payload"

    chunk = backend.generate_next(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "What is shown?"},
        ],
        "solver",
        images=[png],
    )

    request = captured["request"]
    assert request.full_url == "https://example.test/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret-key"
    payload = json.loads(request.data.decode("utf-8"))
    content = payload["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "What is shown?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["timeout"] == 12
    assert chunk.text == "answer"
    assert chunk.request_id == "request-1"
    assert "secret-key" not in json.dumps(chunk.metadata)
