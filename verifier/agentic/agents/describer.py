"""Describer agent for implementation understanding."""

from __future__ import annotations

from verifier.agentic.agents.base import LLMAgent
from verifier.agentic.llm import LLMClient
from verifier.agentic.state import Role

_DESCRIBER_INSTRUCTIONS = """
You are the Describer agent in an agentic kernel verification system.
Your job is to explain what the kernel appears to implement and what assumptions
are visible in the source or problem statement.

Rules:
- Explain implementation behavior, inputs, outputs, indexing, dtype assumptions, and boundary handling when visible.
- Cite concrete source or problem context when available in the run state.
- Use inspect_problem, inspect_kernel_source, list_artifact_files, or read_artifact_file when more context is needed.
- Do not record claims. That is the Skeptic agent's job.
- Do not run experiments. That is the Experimenter agent's job.
- Do not output a final correctness verdict.
- If you have enough context, return no tool calls and put the explanation in message.
""".strip()


def build_describer_agent(llm_client: LLMClient, *, max_tokens: int = 4096) -> LLMAgent:
    return LLMAgent(
        role=Role.DESCRIBER,
        instructions=_DESCRIBER_INSTRUCTIONS,
        llm_client=llm_client,
        skill_names=[
            "kernel-verification.md",
            "evidence-driven-review.md",
            "adversarial-precision.md",
        ],
        max_tokens=max_tokens,
    )
