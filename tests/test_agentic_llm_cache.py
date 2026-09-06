"""Prompt-cache breakpoint placement for the Anthropic client.

Each agent re-sends a byte-identical tools+system prefix on every call of a
run, so a cache breakpoint on the last system block turns that prefix into a
0.1x read after the first call. These tests pin the request shape, since a
silently dropped cache_control costs money without failing anything.
"""
from __future__ import annotations

import pytest

from verifier.agentic.llm import _anthropic_system_blocks, prompt_cache_ttl


def test_default_ttl_is_one_hour() -> None:
    assert prompt_cache_ttl() == "1h"


@pytest.mark.parametrize("value", ["off", "none", "0", "false", "OFF"])
def test_cache_can_be_disabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AGENTIC_PROMPT_CACHE", value)
    assert prompt_cache_ttl() is None
    assert _anthropic_system_blocks("hello") == "hello"


def test_unrecognized_value_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_PROMPT_CACHE", "10 minutes")
    assert prompt_cache_ttl() == "1h"


def test_one_hour_ttl_is_explicit_in_the_block() -> None:
    blocks = _anthropic_system_blocks("system prompt")
    assert blocks == [
        {
            "type": "text",
            "text": "system prompt",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]


def test_five_minute_ttl_omits_the_ttl_field(monkeypatch: pytest.MonkeyPatch) -> None:
    # 5m is the API default; sending it explicitly is rejected.
    monkeypatch.setenv("AGENTIC_PROMPT_CACHE", "5m")
    blocks = _anthropic_system_blocks("system prompt")
    assert blocks == [
        {"type": "text", "text": "system prompt", "cache_control": {"type": "ephemeral"}}
    ]


def test_empty_system_is_not_wrapped() -> None:
    # An empty text block is not a valid request body.
    assert _anthropic_system_blocks("") == ""
