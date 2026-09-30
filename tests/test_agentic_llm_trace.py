"""Raw chat-completion traces, tested entirely through an offline SDK transport."""

from __future__ import annotations

from contextlib import contextmanager
import json

import httpx
import openai
import pytest

from verifier.agentic.llm import build_llm_client
from verifier.agentic import llm_trace


@pytest.fixture(autouse=True)
def isolate_trace_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_LLM_TRACE_DIR", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-secret-key")


@contextmanager
def offline_client(monkeypatch, respond):
    sdk_client = openai.OpenAI
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: sdk_client(
            **kwargs, http_client=http_client, max_retries=0,
        ))
        yield build_llm_client(provider="fireworks")


def completion(*, content='{"verdict":"reject"}', finish_reason="stop", tool_calls=None):
    message = {"role": "assistant", "content": content,
               "reasoning_content": "Provider reasoning that the parsed agent text omits."}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "offline-completion", "object": "chat.completion", "created": 0,
        "model": "accounts/fireworks/models/glm-5p3",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 256, "total_tokens": 356,
                  "completion_tokens_details": {"reasoning_tokens": 250}},
    }


def read(path):
    return json.loads(path.read_text())


def test_full_response_and_request_precede_parsing(tmp_path, monkeypatch):
    root = tmp_path / "llm_calls"
    monkeypatch.setenv("AGENTIC_LLM_TRACE_DIR", str(root))
    observed = []

    def respond(request):
        calls = sorted(root.iterdir())
        call = calls[-1]
        body = json.loads(request.content)
        # This assertion happens before the offline API returns its response.
        assert read(call / "request.json") == body
        assert read(call / "metadata.json")["status"] == "started"
        assert not (call / "response.json").exists()
        observed.append(call)
        return httpx.Response(200, json=completion(finish_reason="length"))

    with offline_client(monkeypatch, respond) as client:
        for _ in range(2):
            assert json.loads(client.call(system="Return JSON.", user="Inspect.", max_tokens=256)) == {
                "verdict": "reject"}

    assert len(set(observed)) == 2
    assert observed == sorted(observed)
    for call in observed:
        response = read(call / "response.json")
        assert response["choices"][0]["finish_reason"] == "length"
        assert response["choices"][0]["message"]["reasoning_content"].startswith("Provider reasoning")
        assert response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 250
        metadata = read(call / "metadata.json")
        assert metadata["status"] == "completed"
        assert metadata["response_saved"] is True
        assert metadata["duration_s"] >= 0
        assert metadata["finished_at"] >= metadata["started_at"]
        assert not (call / "error.json").exists()
    serialized = "\n".join(p.read_text() for p in root.rglob("*.json"))
    assert "offline-secret-key" not in serialized
    assert "authorization" not in serialized.lower()


def test_disabled_trace_does_not_access_filesystem(monkeypatch):
    def fail_path(*args, **kwargs):
        raise AssertionError("Tracing disabled must not access a path")

    monkeypatch.setattr(llm_trace, "Path", fail_path)
    with offline_client(monkeypatch, lambda request: httpx.Response(200, json=completion())) as client:
        assert json.loads(client.call(system="JSON", user="Inspect."))["verdict"] == "reject"


def test_api_failure_saved_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_LLM_TRACE_DIR", str(tmp_path))

    def respond(request):
        return httpx.Response(401, json={"error": {
            "message": "Rejected key offline-secret-key; Authorization: Bearer hidden-token",
            "type": "authentication_error", "code": "invalid_api_key",
        }})

    with offline_client(monkeypatch, respond) as client:
        with pytest.raises(openai.AuthenticationError):
            client.call(system="JSON", user="Inspect.")
    call, = tmp_path.iterdir()
    error = read(call / "error.json")
    assert error["type"] == "openai.AuthenticationError"
    assert error["status_code"] == 401
    assert "[REDACTED]" in error["message"]
    serialized = "\n".join(p.read_text() for p in call.iterdir())
    assert "offline-secret-key" not in serialized
    assert "hidden-token" not in serialized
    metadata = read(call / "metadata.json")
    assert metadata["status"] == "error"
    assert metadata["response_saved"] is False
    assert not (call / "response.json").exists()


def test_malformed_tool_arguments_keep_raw_response(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_LLM_TRACE_DIR", str(tmp_path))
    payload = completion(content=None, finish_reason="tool_calls", tool_calls=[{
        "id": "call_1", "type": "function",
        "function": {"name": "probe", "arguments": "not-json"},
    }])
    tools = [{"name": "probe", "description": "Inspect the kernel.",
              "input_schema": {"type": "object", "properties": {}}}]
    with offline_client(monkeypatch, lambda request: httpx.Response(200, json=payload)) as client:
        with pytest.raises(json.JSONDecodeError):
            client.call(system="JSON", user="Inspect.", tools=tools)
    call, = tmp_path.iterdir()
    assert read(call / "request.json")["tools"][0]["function"]["name"] == "probe"
    assert read(call / "response.json")["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "not-json"
    assert read(call / "error.json")["type"] == "json.decoder.JSONDecodeError"
    metadata = read(call / "metadata.json")
    assert metadata["status"] == "error"
    assert metadata["response_saved"] is True


def test_unwritable_trace_prevents_paid_request(tmp_path, monkeypatch):
    root = tmp_path / "not_a_directory"
    root.write_text("occupied")
    monkeypatch.setenv("AGENTIC_LLM_TRACE_DIR", str(root))

    def respond(request):
        pytest.fail("API called before trace destination was writable")

    with offline_client(monkeypatch, respond) as client:
        with pytest.raises(OSError):
            client.call(system="JSON", user="Inspect.")
