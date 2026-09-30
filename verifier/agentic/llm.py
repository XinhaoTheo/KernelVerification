"""LLM client adapters for agentic verification."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

from .llm_trace import capture_chat_completion

load_dotenv()

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
_DEFAULT_OPENAI_MODEL = "gpt-5"
_DEFAULT_OPENROUTER_MODEL = "z-ai/glm-5.3-flash"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/glm-5p3"
_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_OPENAI_REASONING_EFFORT = "minimal"
# Anthropic prompt-cache TTL for the stable tools+system prefix. "1h" survives a
# slow round (a probe run can put minutes between two turns of the same role);
# "5m" is cheaper to write but misses across those gaps. "off" disables caching.
_DEFAULT_PROMPT_CACHE_TTL = "1h"


class OutputTokenBudgetExhausted(RuntimeError):
    """The run's shared output-token allowance is used up; no request was sent."""


class OutputTokenBudgetAccountingError(RuntimeError):
    """Provider usage cannot support an honest shared-budget measurement."""


@dataclass(slots=True)
class _OutputTokenBudget:
    """One allowance per client, shared by all roles in a sequential run.

    Output usage includes provider-reported reasoning tokens. This is neither
    a dollar budget nor an input-token budget, and reserves nothing for a judge.
    """

    total: int
    used: int = 0
    accounting_error: str | None = None

    def limit(self, requested: int) -> int:
        if self.accounting_error:
            raise OutputTokenBudgetAccountingError(self.accounting_error)
        remaining = self.total - self.used
        if remaining <= 0:
            raise OutputTokenBudgetExhausted(
                f"Shared output-token budget exhausted: used={self.used}, total={self.total}; "
                "no additional API request was sent"
            )
        return min(requested, remaining)

    def record(self, output_tokens: Any) -> None:
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0:
            self.accounting_error = "Provider did not return valid output-token usage; budget accounting stopped"
        else:
            self.used += output_tokens
            if self.used > self.total:
                self.accounting_error = (
                    f"Provider exceeded shared output-token budget: used={self.used}, total={self.total}"
                )
        if self.accounting_error:
            raise OutputTokenBudgetAccountingError(self.accounting_error)


def _output_token_budget_from_env() -> _OutputTokenBudget | None:
    raw = os.getenv("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET")
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET must be a positive integer") from None
    if value <= 0:
        raise ValueError("AGENTIC_TOTAL_OUTPUT_TOKEN_BUDGET must be a positive integer")
    return _OutputTokenBudget(value)


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
    _output_budget: _OutputTokenBudget | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from anthropic import Anthropic

        self._output_budget = _output_token_budget_from_env()
        kwargs: dict[str, Any] = {"timeout": default_timeout_seconds()}
        if self._output_budget is not None:
            # SDK retries would be additional, unrecorded model requests.
            kwargs["max_retries"] = 0
        self._client = Anthropic(**kwargs)

    def call(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        self.last_metrics = None
        if self._output_budget is not None:
            max_tokens = self._output_budget.limit(max_tokens)
        kwargs: dict[str, Any] = {
            "model": self.model or default_model("anthropic"),
            "max_tokens": max_tokens,
            "system": _anthropic_system_blocks(system),
            "messages": [{"role": "user", "content": user}],
        }
        if tools:
            kwargs["tools"] = _anthropic_tool_specs(tools)
            kwargs["tool_choice"] = {"type": "auto"}

        started = time.monotonic()
        response = self._client.messages.create(**kwargs)
        self.last_metrics = _anthropic_call_metrics(response, duration_s=time.monotonic() - started)
        if self._output_budget is not None:
            self._output_budget.record(getattr(getattr(response, "usage", None), "output_tokens", None))

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
    # OpenAI-compatible providers reuse the tool serialization and metrics.
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    use_responses: bool = True
    last_metrics: CallMetrics | None = field(default=None, init=False)
    _client: Any = field(init=False, repr=False)
    _output_budget: _OutputTokenBudget | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._output_budget = _output_token_budget_from_env()
        kwargs: dict[str, Any] = {"timeout": default_timeout_seconds()}
        if self._output_budget is not None:
            kwargs["max_retries"] = 0
        if self.base_url:
            kwargs["base_url"] = self.base_url
        key = os.getenv(self.api_key_env)
        if key:
            kwargs["api_key"] = key
        elif self.api_key_env != "OPENAI_API_KEY":
            raise RuntimeError(
                f"{self.api_key_env} is not set; put it in .env, which is gitignored"
            )
        self._client = OpenAI(**kwargs)

    def call(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        self.last_metrics = None
        if self._output_budget is not None:
            max_tokens = self._output_budget.limit(max_tokens)
        if not self.use_responses or not hasattr(self._client, "responses"):
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
        reasoning_effort = os.getenv("AGENTIC_OPENAI_REASONING_EFFORT")
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        with capture_chat_completion(kwargs) as trace:
            started = time.monotonic()
            response = self._client.chat.completions.create(**kwargs)
            self.last_metrics = _openai_chat_call_metrics(response, duration_s=time.monotonic() - started)
            if trace is not None:
                trace.record_response(response)
            if self._output_budget is not None:
                self._output_budget.record(getattr(getattr(response, "usage", None), "completion_tokens", None))
            message = response.choices[0].message
            text = _chat_message_text(message)
            if not tools:
                return text

            return _serialize_tool_response(
                message=text,
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
        if self._output_budget is not None:
            self._output_budget.record(getattr(getattr(response, "usage", None), "output_tokens", None))

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
    if selected == "openrouter":
        # One gateway, many open-weight models. Opus ran the 32-case benchmark
        # for $88 across both arms; the same token volume on an open model
        # priced at $0.075/$0.25 per M is about $1.22, which is why this exists.
        return OpenAILLMClient(
            model=model,
            base_url=_OPENROUTER_BASE_URL,
            api_key_env="OPENROUTER_API_KEY",
        )
    if selected == "fireworks":
        return OpenAILLMClient(
            model=model or default_model("fireworks"),
            base_url=_FIREWORKS_BASE_URL,
            api_key_env="FIREWORKS_API_KEY",
            # SDK support for Responses does not imply endpoint/model support.
            use_responses=False,
        )
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


def prompt_cache_ttl() -> str | None:
    """Anthropic cache TTL, or None to send no cache_control at all.

    AGENTIC_PROMPT_CACHE accepts "1h" (default), "5m", or "off"/"none"/"0".
    """
    raw = (os.getenv("AGENTIC_PROMPT_CACHE") or _DEFAULT_PROMPT_CACHE_TTL).strip().lower()
    if raw in {"off", "none", "0", "false"}:
        return None
    if raw in {"5m", "1h"}:
        return raw
    return _DEFAULT_PROMPT_CACHE_TTL


def default_provider() -> str:
    return os.getenv("AGENTIC_PROVIDER") or _DEFAULT_PROVIDER


def default_model(provider: str | None = None) -> str:
    explicit = os.getenv("AGENTIC_MODEL")
    if explicit:
        return explicit
    selected = (provider or default_provider()).lower()
    if selected == "fireworks":
        return os.getenv("AGENTIC_FIREWORKS_MODEL") or _DEFAULT_FIREWORKS_MODEL
    if selected == "openrouter":
        return os.getenv("AGENTIC_OPENROUTER_MODEL") or _DEFAULT_OPENROUTER_MODEL
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



def _chat_message_text(message: Any) -> str:
    """The assistant's prose, wherever this provider decided to put it.

    Some OpenAI-compatible endpoints return a reasoning model's prose in a
    separate `reasoning_content` field and leave `content` empty. GLM-5.3 on
    Fireworks does exactly that: a turn that calls a tool comes back with
    content='', reasoning_content holding the entire analysis, and the tool call
    itself intact. Reading only `content` there loses every word the agent
    wrote while keeping the actions it took -- runs still complete, and their
    transcripts are blank.

    Preferring `content` keeps providers that populate it unchanged.
    """
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    return (getattr(message, "reasoning_content", None) or "").strip()


def _serialize_tool_response(*, message: str, tool_calls: list[dict[str, Any]]) -> str:
    """Re-encode a provider's native tool-call result as the agent JSON protocol text.

    Keeping this as the single choke point means parse_agent_response, LLMAgent,
    and every test built against the text protocol stay unchanged regardless of
    which provider or calling convention produced the result.
    """
    return json.dumps({"message": message, "tool_calls": tool_calls})


def _anthropic_system_blocks(system: str) -> Any:
    """Render `system` as content blocks, marking the cacheable prefix.

    An agent's tools and system prompt (instructions + skill documents) are
    byte-identical on every call it makes during a run -- only the run-state
    JSON in the user message changes. A cache breakpoint on the final system
    block therefore caches the whole `tools` + `system` prefix, which is ~25%
    of a typical prompt here, at 0.1x input price on every call after the first.

    Returns the plain string when caching is off or the prompt is empty, so the
    request shape is unchanged in that case.
    """
    ttl = prompt_cache_ttl()
    if not ttl or not system:
        return system
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl != "5m":
        # 5m is the API default and is rejected as an explicit value.
        cache_control["ttl"] = ttl
    return [{"type": "text", "text": system, "cache_control": cache_control}]


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
