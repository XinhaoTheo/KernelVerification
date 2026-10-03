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
    monkeypatch.delenv("AGENTIC_DEBATE_BUDGET_CLOSEOUT", raising=False)
    monkeypatch.delenv("AGENTIC_REQUIRE_CLAIM_SCOPE_FIELDS", raising=False)
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


@pytest.mark.parametrize("consume_as", ["confirmed", "inconclusive", None])
def test_debate_reserves_evidence_review_and_judge_with_real_sdk(monkeypatch, tmp_path, consume_as):
    """Reproduce the 109 failure boundary: a probe exists at 24K tokens used."""
    from verifier.agentic.agents.base import LLMAgent
    from verifier.agentic.ledger import ClaimLedger
    from verifier.agentic.orchestrator import AgenticOrchestrator
    from verifier.agentic.state import Role, RunState, ToolEvent

    monkeypatch.setenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", "32768")
    monkeypatch.setenv("AGENTIC_DEBATE_BUDGET_CLOSEOUT", "1")
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        index = len(requests)
        calls = []
        if index == 4:
            # Even a model attempting a fresh probe cannot execute it in closeout.
            calls.append(("run_python_probe", {"code": "raise AssertionError('must not run')"}))
            if consume_as is not None:
                calls.append(("finalize_probe_evidence", {
                    "event_id": "t1", "supports": consume_as,
                    "summary": "Existing probe interpreted against the contract.",
                    "data": {"ratio": 25.26},
                }))
        elif index == 5:
            calls.append(("record_no_new_claims", {"reason": "Reviewed existing evidence.",
                                                   "reviewed_claims": ["c1"]}))
        elif index == 6:
            calls.append(("record_verdict", {
                "verdict": "reject" if consume_as == "confirmed" else "needs_more_evidence",
                "confidence": 0.8, "decisive_claims": ["c1"] if consume_as == "confirmed" else [],
                "reason": "Existing scoped evidence is decisive." if consume_as == "confirmed"
                          else "The remaining evidence is insufficient; token exhaustion is not a pass.",
            }))
        payload = completion(body["max_tokens"])
        if calls:
            payload["choices"][0]["message"]["tool_calls"] = [
                {"id": f"call_{i}", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(calls)
            ]
        return httpx.Response(200, json=payload)

    with offline_client(monkeypatch, respond) as client:
        agents = [LLMAgent(role, f"You are {role.value}.", client, max_tokens=32768)
                  for role in (Role.DESCRIBER, Role.SKEPTIC, Role.EXPERIMENTER, Role.JUDGE)]
        # Actual SDK calls spend the entire exploration allowance, not a patched counter.
        for _ in range(3):
            agents[0].act(state=RunState(), tools=TOOLS)
        orchestrator = AgenticOrchestrator(run_dir=tmp_path)
        ClaimLedger(orchestrator.state).record_claim(
            statement="The output may exceed the declared error bound.", rationale="Reduction precision.",
            scope="in_scope", scope_rationale="The error bound applies to all legal inputs.",
            scope_evidence=[{"source": "meta.json", "summary": "Bounded input contract."}],
        )
        orchestrator.state.tool_events.append(ToolEvent(
            id="t1", tool="run_claim_probe", args={}, status="ok",
            output={"claim_id": "c1", "exit_code": 0, "json_result": {"ratio": 25.26}},
        ))
        orchestrator.registry.get("run_python_probe").handler = (
            lambda context, args: pytest.fail("closeout executed a new probe"))
        result = orchestrator.run_verification_workflow(
            agents, max_debate_rounds=4, max_claim_rounds=2,
        )
        assert result.stop_reason == "verdict_recorded"
        assert result.rounds_completed == 1
        assert client._output_budget.used == 32768
        assert orchestrator.state.verdict["verdict"] == (
            "reject" if consume_as == "confirmed" else "needs_more_evidence")
        assert orchestrator.state.verdict["output_budget_closeout"]["started_remaining_tokens"] == 8192
        assert [turn.role for turn in orchestrator.state.history] == [
            Role.EXPERIMENTER, Role.SKEPTIC, Role.JUDGE]
        assert orchestrator.state.tool_events[1].output["error_type"] == "CloseoutToolError"
        if consume_as is None:
            assert orchestrator.state.claims[0].status == "inconclusive"
            assert orchestrator.state.verdict["output_budget_closeout"]["unresolved_claims"] == ["c1"]

    assert [body["max_tokens"] for body in requests] == [8192, 8192, 8192, 4096, 1024, 3072]
    assert "budget_evidence_consumption" in requests[3]["messages"][1]["content"]
    assert "final_verdict_required" in requests[5]["messages"][1]["content"]
    assert {tool["function"]["name"] for tool in requests[5]["tools"]} == {"record_verdict"}
    assert all(tool["function"]["name"] not in {"run_python_probe", "run_claim_probe"}
               for tool in requests[3]["tools"])


def test_closeout_option_does_not_cap_solo(monkeypatch):
    from verifier.agentic.agents.base import LLMAgent
    from verifier.agentic.state import RunState

    monkeypatch.setenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", "32768")
    monkeypatch.setenv("AGENTIC_DEBATE_BUDGET_CLOSEOUT", "1")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion(1))

    with offline_client(monkeypatch, respond) as client:
        LLMAgent("solo", "Solo.", client, max_tokens=32768).act(state=RunState(), tools=TOOLS)
    assert requests[0]["max_tokens"] == 32768


def test_explicit_scope_schema_keeps_unknown_legal_and_legacy_default(monkeypatch, tmp_path):
    from verifier.agentic.orchestrator import AgenticOrchestrator
    from verifier.agentic.protocol import AgentResponse
    from verifier.agentic.state import Role, ToolCall
    from verifier.agentic.tools.claims import record_claim_schema

    assert record_claim_schema()["required"] == ["statement", "rationale"]
    monkeypatch.setenv("AGENTIC_REQUIRE_CLAIM_SCOPE_FIELDS", "1")
    schema = record_claim_schema()
    assert set(schema["required"]) == {
        "statement", "rationale", "scope", "scope_rationale", "scope_evidence"}
    orchestrator = AgenticOrchestrator(run_dir=tmp_path)
    outputs = orchestrator.apply_agent_response(role=Role.SKEPTIC, response=AgentResponse(
        message="Scope remains unknown.", tool_calls=[ToolCall("record_claim", {
            "statement": "A stride condition might be relevant.", "rationale": "Inspect scope.",
            "scope": "unknown", "scope_rationale": "", "scope_evidence": [],
        })]))
    assert "error" not in outputs[0]["output"]
    assert orchestrator.state.claims[0].scope == "unknown"


@pytest.mark.parametrize("budget", [None, "8192", "4096"])
def test_closeout_rejects_missing_or_insufficient_shared_budget(monkeypatch, budget):
    monkeypatch.setenv("AGENTIC_DEBATE_BUDGET_CLOSEOUT", "1")
    if budget is not None:
        monkeypatch.setenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET", budget)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: pytest.fail("SDK should not be constructed"))
    with pytest.raises(ValueError, match="requires a total output budget > 8192"):
        build_llm_client(provider="fireworks")
