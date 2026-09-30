"""Shared run budgets exercised through the real SDK with offline HTTP only."""

from contextlib import contextmanager
import json

import httpx
import openai
import pytest

from verifier.agentic.llm import (
    OutputTokenBudgetAccountingError,
    OutputTokenBudgetExhausted,
    build_llm_client,
)


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    monkeypatch.delenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("AGENTIC_LLM_TRACE_DIR", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-budget-test-key")


def completion(output_tokens, *, tool_args=None):
    message = {"role": "assistant", "content": '{"message":"ok","tool_calls":[]}'}
    if tool_args is not None:
        message["tool_calls"] = [{
            "id": "probe_1", "type": "function",
            "function": {"name": "probe", "arguments": tool_args},
        }]
    return {
        "id": "offline-budget-call", "object": "chat.completion", "created": 0,
        "model": "accounts/fireworks/models/glm-5p3",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": output_tokens,
                  "total_tokens": 11 + output_tokens,
                  "completion_tokens_details": {"reasoning_tokens": max(0, output_tokens - 1)}},
    }


@contextmanager
def offline_client(monkeypatch, respond):
    constructor = openai.OpenAI
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: constructor(
            **kwargs, http_client=http_client,
        ))
        yield build_llm_client(provider="fireworks")


TOOLS = [{"name": "probe", "description": "Inspect.",
          "input_schema": {"type": "object", "properties": {}}}]


def test_budget_is_shared_across_roles_and_tool_turns(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", "10")
    monkeypatch.setenv("AGENTIC_LLM_TRACE_DIR", str(tmp_path))
    requests = []
    usage = iter([3, 5, 2])

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(next(usage), tool_args="{}"))

    with offline_client(monkeypatch, respond) as client:
        for role in ("describer", "skeptic", "experimenter"):
            client.call(system=role, user="inspect", tools=TOOLS, max_tokens=8)
        with pytest.raises(OutputTokenBudgetExhausted, match="used=10, total=10"):
            client.call(system="judge", user="decide", max_tokens=8)
        assert client.last_metrics is None  # No stale usage for the unsent call.

    assert [request["max_tokens"] for request in requests] == [8, 7, 2]
    assert [request["messages"][0]["content"] for request in requests] == [
        "describer", "skeptic", "experimenter"]
    calls = sorted(tmp_path.iterdir())
    assert len(calls) == 3
    assert sum(json.loads((call / "response.json").read_text())["usage"]["completion_tokens"]
               for call in calls) == 10
    assert [json.loads((call / "request.json").read_text())["max_tokens"]
            for call in calls] == [8, 7, 2]


def test_no_budget_preserves_per_call_limits(monkeypatch):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(7))

    with offline_client(monkeypatch, respond) as client:
        for _ in range(3):
            client.call(system="solo", user="inspect", max_tokens=8)
        assert client._client.max_retries == 2
    assert [request["max_tokens"] for request in requests] == [8, 8, 8]


def test_malformed_tool_response_still_spends_actual_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", "10")
    monkeypatch.setenv("AGENTIC_LLM_TRACE_DIR", str(tmp_path))
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(
            7 if len(requests) == 1 else 3, tool_args="not-json" if len(requests) == 1 else "{}"))

    with offline_client(monkeypatch, respond) as client:
        with pytest.raises(json.JSONDecodeError):
            client.call(system="skeptic", user="inspect", tools=TOOLS, max_tokens=10)
        client.call(system="experimenter", user="inspect", tools=TOOLS, max_tokens=10)
        with pytest.raises(OutputTokenBudgetExhausted):
            client.call(system="judge", user="decide", max_tokens=10)
    assert [request["max_tokens"] for request in requests] == [10, 3]
    first = sorted(tmp_path.iterdir())[0]
    assert json.loads((first / "metadata.json").read_text())["response_saved"] is True
    assert json.loads((first / "metadata.json").read_text())["status"] == "error"


def test_missing_usage_stops_budget_accounting_and_keeps_response(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", "10")
    monkeypatch.setenv("AGENTIC_LLM_TRACE_DIR", str(tmp_path))
    requests = []

    def respond(request):
        requests.append(request)
        response = completion(1)
        response.pop("usage")
        return httpx.Response(200, json=response)

    with offline_client(monkeypatch, respond) as client:
        for _ in range(2):
            with pytest.raises(OutputTokenBudgetAccountingError, match="valid output-token usage"):
                client.call(system="solo", user="inspect", max_tokens=10)
    assert len(requests) == 1
    call, = tmp_path.iterdir()
    assert json.loads((call / "metadata.json").read_text())["response_saved"] is True


def test_budgeted_client_does_not_retry_transport_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", "10")
    monkeypatch.setenv("AGENTIC_LLM_TRACE_DIR", str(tmp_path))
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "offline failure"}})

    with offline_client(monkeypatch, respond) as client:
        with pytest.raises(openai.InternalServerError):
            client.call(system="solo", user="inspect", max_tokens=10)
        assert client.last_metrics is None
    assert len(requests) == 1
    call, = tmp_path.iterdir()
    assert json.loads((call / "metadata.json").read_text())["status"] == "error"


@pytest.mark.parametrize("invalid", ["0", "-10", "abc", "1.5"])
def test_invalid_budget_fails_before_constructing_client(monkeypatch, invalid):
    monkeypatch.setenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", invalid)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: pytest.fail("SDK should not be constructed"))
    with pytest.raises(ValueError, match="positive integer"):
        build_llm_client(provider="fireworks")
