"""Tests for the OpenAI-compatible Teacher backend.

All requests go to a local ``http.server`` stub; no real Teacher is contacted.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from tools.local_sft_builder.budget import TeacherRequestBudget
from tools.local_sft_builder.manifest import build_manifest
from tools.local_sft_builder.teacher_backend import (
    SamplingConfig,
    TeacherBackend,
    TeacherBackendError,
    build_request_messages,
    image_data_url,
)


Responder = Callable[[], tuple[int, Any, float]]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        state = self.server.stub_state  # type: ignore[attr-defined]
        state["requests"].append(
            {
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "payload": json.loads(raw.decode("utf-8")),
            }
        )
        status, body, delay = state["responder"]()
        if delay:
            time.sleep(delay)
        encoded = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("x-request-id", "stub-request-id")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: Any) -> None:  # silence the stub
        return


class StubTeacher:
    """A local OpenAI-compatible endpoint with observable requests."""

    def __init__(self, responder: Responder | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responder = responder or self.default_responder
        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._server.stub_state = {"requests": self.requests, "responder": self._call}
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @staticmethod
    def default_responder() -> tuple[int, Any, float]:
        return (
            200,
            {
                "id": "chatcmpl-stub",
                "model": "stub-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "stub answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
            0.0,
        )

    def _call(self) -> tuple[int, Any, float]:
        return self.responder()

    def __enter__(self) -> "StubTeacher":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"


def _write_image(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


# --------------------------------------------------------------------------
# Request construction
# --------------------------------------------------------------------------


def test_generate_posts_openai_compatible_payload(tmp_path: Path) -> None:
    with StubTeacher() as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        response = backend.generate(
            role="natural",
            messages=[{"role": "user", "content": "hello"}],
            sampling=SamplingConfig(
                temperature=0.3, top_p=0.8, max_tokens=64, timeout=10.0, seed=7
            ),
        )

    assert len(stub.requests) == 1
    request = stub.requests[0]
    assert request["path"] == "/v1/chat/completions"
    payload = request["payload"]
    assert payload["model"] == "stub-model"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.8
    assert payload["max_tokens"] == 64
    assert payload["stream"] is False
    assert payload["seed"] == 7

    assert response.text == "stub answer"
    assert response.model == "stub-model"
    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 5, "completion_tokens": 2}
    assert response.logical_request_id == "stub-request-id"


def test_images_are_expanded_in_placeholder_order(tmp_path: Path) -> None:
    first = _write_image(tmp_path, "first.png", b"first-image-bytes")
    second = _write_image(tmp_path, "second.jpg", b"second-image-bytes")

    with StubTeacher() as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        backend.generate(
            role="natural",
            messages=[{"role": "user", "content": "before <image> middle <image> after"}],
            images=[str(first), str(second)],
        )

    blocks = stub.requests[0]["payload"]["messages"][0]["content"]
    assert [block["type"] for block in blocks] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert [block["text"] for block in blocks if block["type"] == "text"] == [
        "before ",
        " middle ",
        " after",
    ]
    image_urls = [block["image_url"]["url"] for block in blocks if block["type"] == "image_url"]
    assert image_urls[0] == image_data_url(first)
    assert image_urls[1] == image_data_url(second)
    assert image_urls[0].startswith("data:image/png;base64,")
    assert image_urls[1].startswith("data:image/jpeg;base64,")


def test_image_placeholder_is_kept_across_turns(tmp_path: Path) -> None:
    image = _write_image(tmp_path, "chart.png", b"chart-bytes")

    messages = build_request_messages(
        [
            {"role": "user", "content": "<image>\nQuestion: value?"},
            {"role": "assistant", "content": "step one"},
            {"role": "user", "content": "\n[Code Execution Result]\nOutput: 4\n"},
        ],
        [str(image)],
    )

    assert isinstance(messages[0]["content"], list)
    assert messages[1] == {"role": "assistant", "content": "step one"}
    assert messages[2]["content"].startswith("\n[Code Execution Result]")


def test_placeholder_and_image_count_must_match(tmp_path: Path) -> None:
    image = _write_image(tmp_path, "one.png", b"one")

    with StubTeacher() as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        with pytest.raises(TeacherBackendError):
            backend.generate(
                role="natural",
                messages=[{"role": "user", "content": "no placeholder"}],
                images=[str(image)],
            )
        with pytest.raises(TeacherBackendError):
            backend.generate(
                role="natural",
                messages=[{"role": "user", "content": "<image> and <image>"}],
                images=[str(image)],
            )

    # Nothing may reach the endpoint when the binding is invalid.
    assert stub.requests == []


def test_unreadable_image_raises_before_request(tmp_path: Path) -> None:
    with StubTeacher() as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        with pytest.raises(TeacherBackendError):
            backend.generate(
                role="natural",
                messages=[{"role": "user", "content": "<image>"}],
                images=[str(tmp_path / "absent.png")],
            )

    assert stub.requests == []


# --------------------------------------------------------------------------
# Failure mapping
# --------------------------------------------------------------------------


def test_timeout_raises_timeout_error_without_retry() -> None:
    with StubTeacher(lambda: (200, {"choices": []}, 2.0)) as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        with pytest.raises(TimeoutError):
            backend.generate(
                role="natural",
                messages=[{"role": "user", "content": "slow"}],
                sampling=SamplingConfig(timeout=0.3),
            )

    assert len(stub.requests) == 1


def test_http_error_raises_backend_error() -> None:
    with StubTeacher(lambda: (500, {"error": "boom"}, 0.0)) as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        with pytest.raises(TeacherBackendError) as excinfo:
            backend.generate(role="natural", messages=[{"role": "user", "content": "hi"}])

    assert "500" in str(excinfo.value)


def test_non_json_response_raises_backend_error() -> None:
    with StubTeacher(lambda: (200, b"<html>not json</html>", 0.0)) as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        with pytest.raises(TeacherBackendError):
            backend.generate(role="natural", messages=[{"role": "user", "content": "hi"}])


def test_response_without_choices_raises_backend_error() -> None:
    with StubTeacher(lambda: (200, {"model": "stub-model"}, 0.0)) as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        with pytest.raises(TeacherBackendError):
            backend.generate(role="natural", messages=[{"role": "user", "content": "hi"}])


def test_unreachable_endpoint_raises_backend_error() -> None:
    # Bind then close a port so the connection is actively refused.  A port such
    # as :1 can simply hang on Windows and surface as a timeout instead.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()

    backend = TeacherBackend(
        base_url=f"http://127.0.0.1:{closed_port}/v1",
        model="stub-model",
    )
    with pytest.raises(TeacherBackendError):
        backend.generate(
            role="natural",
            messages=[{"role": "user", "content": "hi"}],
            sampling=SamplingConfig(timeout=5.0),
        )


# --------------------------------------------------------------------------
# Credential handling
# --------------------------------------------------------------------------


def test_credential_comes_from_env_and_is_never_leaked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-local-test-credential-value"
    monkeypatch.setenv("LOCAL_TEACHER_CREDENTIAL", secret)

    with StubTeacher(lambda: (500, {"error": "boom"}, 0.0)) as stub:
        backend = TeacherBackend(
            base_url=stub.base_url,
            model="stub-model",
            api_key_env="LOCAL_TEACHER_CREDENTIAL",
        )
        assert stub.requests == []
        with pytest.raises(TeacherBackendError) as excinfo:
            backend.generate(role="natural", messages=[{"role": "user", "content": "hi"}])

    assert stub.requests[0]["headers"]["authorization"] == f"Bearer {secret}"
    assert secret not in str(excinfo.value)
    assert secret not in json.dumps(backend.describe())


def test_no_credential_sends_no_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_TEACHER_CREDENTIAL", raising=False)

    with StubTeacher() as stub:
        backend = TeacherBackend(
            base_url=stub.base_url,
            model="stub-model",
            api_key_env="LOCAL_TEACHER_CREDENTIAL",
        )
        backend.generate(role="natural", messages=[{"role": "user", "content": "hi"}])

    assert "authorization" not in stub.requests[0]["headers"]
    assert backend.describe()["teacher_credential_configured"] is False


def test_describe_is_manifest_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_TEACHER_CREDENTIAL", "sk-local-test-credential-value")

    backend = TeacherBackend(
        base_url="http://127.0.0.1:8000/v1",
        model="qwen3.8-27b",
        api_key_env="LOCAL_TEACHER_CREDENTIAL",
    )

    manifest = build_manifest(
        run_id="run-1",
        base_sha="a" * 40,
        extra=backend.describe(),
    )

    assert manifest["teacher_model"] == "qwen3.8-27b"
    assert manifest["teacher_credential_configured"] is True


def test_sampling_config_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        SamplingConfig(temperature=-0.1)
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingConfig(max_tokens=0)
    with pytest.raises(ValueError):
        SamplingConfig(timeout=0.0)


# --------------------------------------------------------------------------
# Ledger integration
# --------------------------------------------------------------------------


def test_budget_records_timeout_and_consumes_one_slot(tmp_path: Path) -> None:
    with StubTeacher(lambda: (200, {"choices": []}, 2.0)) as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        budget = TeacherRequestBudget(str(tmp_path / "ledger.sqlite3"), "task-1", limit=32)
        result = budget.execute(
            role="natural",
            backend=backend.generate_from_payload,
            payload={
                "role": "natural",
                "messages": [{"role": "user", "content": "hi"}],
                "sampling": SamplingConfig(timeout=0.3),
            },
        )

    assert result.status == "timeout"
    stats = budget.stats()
    assert stats.consumed_slot == 1
    assert stats.timeout == 1
    assert stats.pending == 0
    assert len(stub.requests) == 1


def test_budget_records_other_failed_on_backend_error(tmp_path: Path) -> None:
    with StubTeacher(lambda: (503, {"error": "unavailable"}, 0.0)) as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        budget = TeacherRequestBudget(str(tmp_path / "ledger.sqlite3"), "task-1", limit=32)
        result = budget.execute(
            role="natural",
            backend=backend.generate_from_payload,
            payload={"role": "natural", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert result.status == "other_failed"
    stats = budget.stats()
    assert stats.consumed_slot == 1
    assert stats.other_failed == 1
    assert stats.pending == 0


def test_budget_records_successful_response(tmp_path: Path) -> None:
    with StubTeacher() as stub:
        backend = TeacherBackend(base_url=stub.base_url, model="stub-model")
        budget = TeacherRequestBudget(str(tmp_path / "ledger.sqlite3"), "task-1", limit=32)
        result = budget.execute(
            role="natural",
            backend=backend.generate_from_payload,
            payload={"role": "natural", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert result.status == "successful"
    assert result.value.text == "stub answer"
    stats = budget.stats()
    assert stats.successful == 1
    assert stats.pending == 0


def test_completions_url_requires_the_openai_api_root() -> None:
    """The base URL must include ``/v1``.

    A real ``vllm serve`` mounts chat completions at ``/v1/chat/completions``,
    so a base URL without ``/v1`` yields a 404 rather than a request. The
    ``--teacher-base-url`` default therefore has to carry it.
    """

    assert (
        TeacherBackend(base_url="http://127.0.0.1:8000/v1", model="m").completions_url
        == "http://127.0.0.1:8000/v1/chat/completions"
    )
    # A trailing slash must not double up.
    assert (
        TeacherBackend(base_url="http://127.0.0.1:8000/v1/", model="m").completions_url
        == "http://127.0.0.1:8000/v1/chat/completions"
    )
    # Passing the full path is idempotent.
    assert (
        TeacherBackend(
            base_url="http://127.0.0.1:8000/v1/chat/completions", model="m"
        ).completions_url
        == "http://127.0.0.1:8000/v1/chat/completions"
    )


def test_teacher_base_url_default_carries_the_v1_root() -> None:
    from tools.local_sft_builder.run_real_builder import DEFAULT_TEACHER_BASE_URL

    assert DEFAULT_TEACHER_BASE_URL.endswith("/v1")
    assert (
        TeacherBackend(base_url=DEFAULT_TEACHER_BASE_URL, model="m").completions_url
        == f"{DEFAULT_TEACHER_BASE_URL}/chat/completions"
    )
