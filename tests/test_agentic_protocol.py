from __future__ import annotations

import pytest

from verifier.agentic.protocol import ProtocolError, parse_agent_response


def test_parse_agent_response_accepts_tool_calls() -> None:
    response = parse_agent_response(
        '{"message":"inspect first","tool_calls":[{"tool":"inspect_problem","args":{"entry":"toy"}}]}'
    )

    assert response.message == "inspect first"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool == "inspect_problem"
    assert response.tool_calls[0].args == {"entry": "toy"}


def test_parse_agent_response_rejects_bad_json() -> None:
    with pytest.raises(ProtocolError, match="not valid JSON"):
        parse_agent_response("not json")


def test_parse_agent_response_rejects_bad_tool_call_shape() -> None:
    with pytest.raises(ProtocolError, match=r"tool_calls\[0\]\.args"):
        parse_agent_response('{"message":"x","tool_calls":[{"tool":"load_artifact","args":[]}]}' )


def test_parse_agent_response_accepts_fenced_json() -> None:
    response = parse_agent_response(
        "```json\n{\"message\":\"ok\",\"tool_calls\":[]}\n```"
    )

    assert response.message == "ok"
    assert response.tool_calls == []
