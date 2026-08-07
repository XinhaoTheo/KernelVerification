"""Base LLM agent wrapper for JSON tool-call agents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from verifier.agentic.llm import LLMClient
from verifier.agentic.protocol import AgentResponse, ProtocolError, parse_agent_response
from verifier.agentic.state import ClaimStatus, JsonValue, Role, RunState

_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


@dataclass(slots=True)
class LLMAgent:
    role: Role | str
    instructions: str
    llm_client: LLMClient
    skill_names: list[str] = field(default_factory=list)
    max_tokens: int = 4096

    def act(self, *, state: RunState, tools: list[dict[str, JsonValue]]) -> AgentResponse:
        text = self.llm_client.call(
            system=self._build_system_prompt(),
            user=self._build_user_prompt(state=state, tools=tools),
            tools=tools,
            max_tokens=self.max_tokens,
        )
        try:
            return parse_agent_response(text)
        except ProtocolError as exc:
            raise ProtocolError(f"{exc} | raw_response={_snippet(text)!r}") from exc

    def _build_system_prompt(self) -> str:
        parts = [
            self.instructions.strip(),
            "",
            "Tools are provided for this turn through the tool-calling mechanism. To take any "
            "action (record a claim, run a probe, update state, request a verdict, etc.) you "
            "must call the tool directly through that mechanism.",
            "Do not describe a tool call in your message text, and do not write it out as a "
            "JSON object yourself -- a tool call written as text is not executed and has no "
            "effect. Do not claim a tool was run unless it appears in the ledger.",
            "Your message should be a short natural-language explanation of what you are doing "
            "and why, separate from any tool call you make.",
        ]
        skills = _load_skills(self.skill_names)
        if skills:
            parts.extend(["", "=== Skills ===", skills])
        return "\n".join(parts)

    def _build_user_prompt(self, *, state: RunState, tools: list[dict[str, JsonValue]]) -> str:
        return "\n".join(
            [
                "=== Current Run State ===",
                json.dumps(_state_for_prompt(state), indent=2, sort_keys=True, default=str),
                "",
                "=== Available Tools ===",
                json.dumps(tools, indent=2, sort_keys=True, default=str),
                "",
                "Return your next action using the JSON protocol.",
            ]
        )


def _load_skills(skill_names: list[str]) -> str:
    sections = []
    for name in skill_names:
        path = _DEFAULT_SKILLS_DIR / name
        if path.exists():
            sections.append(f"--- {name} ---\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(sections)


def _state_for_prompt(state: RunState) -> dict[str, JsonValue]:
    artifact = dict(state.artifact or {})
    for key in ("kernel_code", "test_code"):
        if isinstance(artifact.get(key), str):
            artifact[key] = _truncate(str(artifact[key]), 12000)
    return cast(dict[str, JsonValue], {
        "entry": state.entry,
        "artifact": artifact,
        "history": [turn.to_dict() for turn in state.history[-6:]],
        "description_model": state.description_model.to_dict(),
        "open_description_tasks": [
            task.to_dict()
            for task in state.description_tasks
            if _status_value(task.status) == "open"
        ],
        "recent_description_updates": [update.to_dict() for update in state.description_updates[-5:]],
        "tool_events": [event.to_dict() for event in state.tool_events[-12:]],
        "claims": [claim.to_dict() for claim in state.claims],
        "claim_coverage": _claim_coverage(state),
        "convergence": state.convergence,
        "skeptic_review": state.skeptic_review,
    })


def _claim_coverage(state: RunState) -> dict[str, JsonValue]:
    open_claim_ids = [
        claim.id
        for claim in state.claims
        if _status_value(claim.status) == ClaimStatus.OPEN.value
    ]
    return cast(dict[str, JsonValue], {
        "open_claim_ids": open_claim_ids,
        "all_open_claims_have_evidence": not open_claim_ids,
    })


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


def _snippet(text: str, *, head: int = 1500, tail: int = 1500) -> str:
    """Head+tail snippet for debugging a malformed raw LLM response.

    Truncation-caused parse errors (e.g. "Unterminated string") are visible
    near the end; prose-wrapping errors ("Expecting value") are visible near
    the start. Keeping both is more useful here than a single-sided cut.
    """
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n...[{len(text) - head - tail} chars omitted]...\n{text[-tail:]}"
