"""LLM client adapters for agentic verification."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
_DEFAULT_OPENAI_MODEL = "gpt-5"
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_OPENAI_REASONING_EFFORT = "minimal"


class LLMClient(Protocol):
    def call(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
        """Return raw model text. The caller parses the agent protocol."""


@dataclass(slots=True)
class AnthropicLLMClient:
    model: str | None = None
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(timeout=default_timeout_seconds())

    def call(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
        response = self._client.messages.create(
            model=self.model or default_model("anthropic"),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


@dataclass(slots=True)
class OpenAILLMClient:
    model: str | None = None
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(timeout=default_timeout_seconds())

    def call(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
        if not hasattr(self._client, "responses"):
            response = self._client.chat.completions.create(
                model=self.model or default_model("openai"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""

        response = self._client.responses.create(
            model=self.model or default_model("openai"),
            instructions=system,
            input=user,
            max_output_tokens=max_tokens,
            reasoning={"effort": default_openai_reasoning_effort()},
            text={"format": {"type": "json_object"}},
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


def _extract_openai_text(response) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "".join(parts)
