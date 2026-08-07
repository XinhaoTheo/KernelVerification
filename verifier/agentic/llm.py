"""LLM client adapters for agentic verification."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
_DEFAULT_OPENAI_MODEL = "gpt-5"
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_OPENAI_REASONING_EFFORT = "minimal"


@dataclass(slots=True)
class CallMetrics:
    """Timing and token usage for one LLM API call, for cost/latency accounting."""

    duration_s: float
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


class LLMClient(Protocol):
    def call(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Return raw model text shaped like the agent JSON protocol.

        When `tools` is given, implementations should prefer the provider's
        native tool-calling so tool_call args are schema-validated by the
        provider instead of hand-parsed from free text, then re-serialize the
        result into the same `{"message": ..., "tool_calls": [...]}` text
        shape so callers do not need to know which path was used.

        Implementations should set `self.last_metrics` (a `CallMetrics`) after
        every call so callers can read timing/token usage without changing
        this return type. Callers must treat it as optional (`getattr(...,
        "last_metrics", None)`) since test doubles are not required to set it.
        """
        ...


@dataclass(slots=True)
class AnthropicLLMClient:
    model: str | None = None
    last_metrics: CallMetrics | None = field(default=None, init=False)
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(timeout=default_timeout_seconds())

    def call(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model or default_model("anthropic"),
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if tools:
            kwargs["tools"] = _anthropic_tool_specs(tools)
            kwargs["tool_choice"] = {"type": "auto"}

        started = time.monotonic()
        response = self._client.messages.create(**kwargs)
        self.last_metrics = _anthropic_call_metrics(response, duration_s=time.monotonic() - started)

        if tools:
            return _serialize_tool_response(
                message="".join(
                    block.text for block in response.content if getattr(block, "type", None) == "text"
                ),
                tool_calls=[
                    {"tool": block.name, "args": dict(block.input)}
                    for block in response.content
                    if getattr(block, "type", None) == "tool_use"
                ],
            )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


@dataclass(slots=True)
class OpenAILLMClient:
    model: str | None = None
    last_metrics: CallMetrics | None = field(default=None, init=False)
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(timeout=default_timeout_seconds())

    def call(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        if not hasattr(self._client, "responses"):
            return self._call_chat_completions(system=system, user=user, tools=tools, max_tokens=max_tokens)
        return self._call_responses(system=system, user=user, tools=tools, max_tokens=max_tokens)

    def _call_chat_completions(
        self, *, system: str, user: str, tools: list[dict[str, Any]] | None, max_tokens: int
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model or default_model("openai"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _openai_chat_tool_specs(tools)
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        response = self._client.chat.completions.create(**kwargs)
        self.last_metrics = _openai_chat_call_metrics(response, duration_s=time.monotonic() - started)
        message = response.choices[0].message
        if not tools:
            return message.content or ""

        return _serialize_tool_response(
            message=message.content or "",
            tool_calls=[
                {"tool": call.function.name, "args": json.loads(call.function.arguments or "{}")}
                for call in (message.tool_calls or [])
            ],
        )

    def _call_responses(
        self, *, system: str, user: str, tools: list[dict[str, Any]] | None, max_tokens: int
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model or default_model("openai"),
            "instructions": system,
            "input": user,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": default_openai_reasoning_effort()},
        }
        if tools:
            kwargs["tools"] = _openai_responses_tool_specs(tools)
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["text"] = {"format": {"type": "json_object"}}

        started = time.monotonic()
        response = self._client.responses.create(**kwargs)
        self.last_metrics = _openai_responses_call_metrics(response, duration_s=time.monotonic() - started)

        if tools:
            return _serialize_tool_response(
                message=_extract_openai_text(response),
                tool_calls=_extract_openai_function_calls(response),
            )

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        extracted = _extract_openai_text(response)
        if extracted:
            return extracted
        incomplete = getattr(response, "incomplete_details", None)
        status = getattr(response, "status", None)
        raise RuntimeError(f"OpenAI response did not contain text; status={status!r}, incomplete_details={incomplete!r}")


def build_llm_client(*, provider: str | None = None, model: str | None = None) -> LLMClient:
    selected = (provider or default_provider()).lower()
    if selected == "anthropic":
        return AnthropicLLMClient(model=model)
    if selected in {"openai", "chatgpt"}:
        return OpenAILLMClient(model=model)
    raise ValueError(f"unsupported LLM provider: {provider}")


def default_openai_reasoning_effort() -> str:
    return os.getenv("AGENTIC_OPENAI_REASONING_EFFORT") or _DEFAULT_OPENAI_REASONING_EFFORT


def default_timeout_seconds() -> float:
    raw = os.getenv("AGENTIC_LLM_TIMEOUT_SECONDS")
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_SECONDS
    return max(1.0, value)


def default_provider() -> str:
    return os.getenv("AGENTIC_PROVIDER") or _DEFAULT_PROVIDER


def default_model(provider: str | None = None) -> str:
    explicit = os.getenv("AGENTIC_MODEL")
    if explicit:
        return explicit
    selected = (provider or default_provider()).lower()
    if selected in {"openai", "chatgpt"}:
        return os.getenv("AGENTIC_OPENAI_MODEL") or _DEFAULT_OPENAI_MODEL
    return os.getenv("AGENTIC_ANTHROPIC_MODEL") or _DEFAULT_ANTHROPIC_MODEL


def _anthropic_call_metrics(response, *, duration_s: float) -> CallMetrics:
    usage = response.usage
    return CallMetrics(
        duration_s=duration_s,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
    )


def _openai_chat_call_metrics(response, *, duration_s: float) -> CallMetrics:
    usage = getattr(response, "usage", None)
    return CallMetrics(
        duration_s=duration_s,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _openai_responses_call_metrics(response, *, duration_s: float) -> CallMetrics:
    usage = getattr(response, "usage", None)
    return CallMetrics(
        duration_s=duration_s,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )


def _serialize_tool_response(*, message: str, tool_calls: list[dict[str, Any]]) -> str:
    """Re-encode a provider's native tool-call result as the agent JSON protocol text.

    Keeping this as the single choke point means parse_agent_response, LLMAgent,
    and every test built against the text protocol stay unchanged regardless of
    which provider or calling convention produced the result.
    """
    return json.dumps({"message": message, "tool_calls": tool_calls})


def _anthropic_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": tool["name"], "description": tool["description"], "input_schema": tool["input_schema"]}
        for tool in tools
    ]


def _openai_chat_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


def _openai_responses_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
        for tool in tools
    ]


def _extract_openai_text(response) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)


def _extract_openai_function_calls(response) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        name = getattr(item, "name", None)
        if not name:
            continue
        raw_args = getattr(item, "arguments", None) or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
        calls.append({"tool": name, "args": args})
    return calls
