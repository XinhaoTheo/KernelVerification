"""Exercise Fireworks through the real SDK with an offline HTTP transport."""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from verifier.agentic.llm import build_llm_client
from verifier.agentic.tools.execution import _strip_sensitive_env
from verifier.agentic_run import main as agentic_main


@pytest.fixture
def fireworks_http(monkeypatch):
    requests = []
    message = {"role": "assistant", "content": '{"message":"ok","tool_calls":[]}'}

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-fireworks-call", "object": "chat.completion", "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        })

    sdk_client = openai.OpenAI
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: sdk_client(
            **kwargs, http_client=http_client,
        ))
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-unrelated-openai-key")
        yield requests, message


@pytest.mark.parametrize("with_tools", [False, True])
def test_fireworks_request_and_response(fireworks_http, with_tools):
    requests, message = fireworks_http
    tools = None
    if with_tools:
        tools = [{"name": "record_claim", "description": "Record a claim.",
                  "input_schema": {"type": "object", "properties": {
                      "statement": {"type": "string"}}, "required": ["statement"]}}]
        message.update(content=None, tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "record_claim", "arguments": '{"statement":"Check boundaries"}'},
        }])

    client = build_llm_client(provider="fireworks")
    result = json.loads(client.call(system="Return JSON.", user="Inspect the kernel.",
                                    tools=tools, max_tokens=256))

    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-fireworks-key"
    body = json.loads(requests[0].content)
    assert body["model"] == "accounts/fireworks/models/glm-5p3"
    assert body["max_tokens"] == 256
    assert "reasoning" not in body
    if with_tools:
        assert body["tools"][0]["function"]["parameters"] == tools[0]["input_schema"]
        assert result["tool_calls"] == [{"tool": "record_claim", "args": {"statement": "Check boundaries"}}]
    else:
        assert body["response_format"] == {"type": "json_object"}
        assert result == {"message": "ok", "tool_calls": []}
    assert client.last_metrics.input_tokens == 12
    assert client.last_metrics.output_tokens == 7


@pytest.mark.parametrize("global_model,explicit_model,expected", [
    (None, None, "provider-model"),
    ("global-model", None, "global-model"),
    ("global-model", "cli-model", "cli-model"),
])
def test_fireworks_model_precedence(fireworks_http, monkeypatch, global_model, explicit_model, expected):
    monkeypatch.setenv("AGENTIC_FIREWORKS_MODEL", "provider-model")
    if global_model:
        monkeypatch.setenv("AGENTIC_MODEL", global_model)
    build_llm_client(provider="fireworks", model=explicit_model).call(system="JSON", user="Test")
    assert json.loads(fireworks_http[0][0].content)["model"] == expected


def test_fireworks_requires_its_own_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-unrelated-openai-key")
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY is not set"):
        build_llm_client(provider="fireworks")


def test_fireworks_cli_reaches_chat_endpoint(tmp_path, fireworks_http, capsys):
    entry = tmp_path / "dataset" / "toy"
    entry.mkdir(parents=True)
    (entry / "meta.json").write_text(json.dumps({"name": "toy", "passed": True, "status": "passed", "rounds": 1}))
    (entry / "problem.txt").write_text("Add one.\n")
    (entry / "kernel.py").write_text("def kernel(x): return x + 1\n")
    (entry / "test.py").write_text("def test(): pass\n")
    assert agentic_main([
        "toy", "--dataset-dir", str(tmp_path / "dataset"), "--run-dir", str(tmp_path / "run"),
        "--provider", "fireworks", "--agent", "describer", "--max-debate-rounds", "1",
    ]) == 0
    assert "provider: fireworks" in capsys.readouterr().out
    assert fireworks_http[0]
    assert all(r.url.path == "/inference/v1/chat/completions" for r in fireworks_http[0])
    assert (tmp_path / "run" / "run.json").exists()


def test_fireworks_key_is_removed_from_probe_environment():
    assert _strip_sensitive_env({"FIREWORKS_API_KEY": "test-key", "PATH": "/bin"}) == {"PATH": "/bin"}
