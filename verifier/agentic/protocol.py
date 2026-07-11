"""Provider-independent agent-to-orchestrator JSON protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .state import JsonValue, ToolCall


class ProtocolError(ValueError):
    """Raised when an agent response does not match the tool-call protocol."""


@dataclass(slots=True)
class AgentResponse:
    message: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "message": self.message,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


def parse_agent_response(text: str) -> AgentResponse:
    """Parse the JSON protocol used between agents and the orchestrator.

    Expected shape:
        {"message": "...", "tool_calls": [{"tool": "...", "args": {...}}]}
    """
    json_text = _extract_json_text(text)
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"agent response is not valid JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ProtocolError("agent response must be a JSON object")

    message = raw.get("message", "")
    if not isinstance(message, str):
        raise ProtocolError("agent response field 'message' must be a string")

    raw_tool_calls = raw.get("tool_calls", [])
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise ProtocolError("agent response field 'tool_calls' must be a list")

    tool_calls = [_parse_tool_call(item, index) for index, item in enumerate(raw_tool_calls)]
    return AgentResponse(message=message, tool_calls=tool_calls)


def _parse_tool_call(raw: Any, index: int) -> ToolCall:
    if not isinstance(raw, dict):
        raise ProtocolError(f"tool_calls[{index}] must be an object")

    tool = raw.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        raise ProtocolError(f"tool_calls[{index}].tool must be a non-empty string")

    args = raw.get("args", {})
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ProtocolError(f"tool_calls[{index}].args must be an object")

    return ToolCall(tool=tool, args=args)


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped
