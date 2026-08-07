"""Provider-independent agent-to-orchestrator JSON protocol."""

from __future__ import annotations

import json
import re
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
    """Best-effort extraction of one JSON object from a raw model response.

    Anthropic's API has no JSON-mode enforcement (unlike OpenAI's
    response_format=json_object), so the model sometimes wraps the object in
    commentary or a fenced block that doesn't span the whole response. Try,
    in order: a fenced block anywhere in the text, then the first balanced
    top-level {...} anywhere in the text. Fall back to the stripped text
    so genuinely invalid input still raises a normal ProtocolError.
    """
    stripped = text.strip()

    fenced = _find_fenced_block(stripped)
    if fenced is not None:
        stripped = fenced

    balanced = _find_balanced_object(stripped)
    if balanced is not None:
        return balanced
    return stripped


def _find_fenced_block(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _find_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
