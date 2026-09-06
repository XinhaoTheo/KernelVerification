"""Shared pytest fixtures.

`verifier.agentic.llm` calls `load_dotenv()` at import time, so a developer's
local .env leaks into `os.environ` for the whole test session. Any default that
reads the environment (AGENTIC_MAX_ROUNDS, AGENTIC_MODEL, ...) would then make
test outcomes depend on that file. Clear those for every test so the suite runs
against the documented defaults instead.
"""
from __future__ import annotations

import pytest

_AGENTIC_ENV_VARS = (
    "AGENTIC_MAX_ROUNDS",
    "AGENTIC_MODEL",
    "AGENTIC_PROVIDER",
    "AGENTIC_ANTHROPIC_MODEL",
    "AGENTIC_OPENAI_MODEL",
    "AGENTIC_PROBE_SANDBOX",
    "AGENTIC_PROBE_MEMORY_MAX_MB",
    "AGENTIC_PROBE_CPU_QUOTA_PCT",
    "AGENTIC_PROMPT_CACHE",
)


@pytest.fixture(autouse=True)
def _isolate_agentic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _AGENTIC_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
