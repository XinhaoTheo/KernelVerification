"""Base LLM agent wrapper for JSON tool-call agents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from verifier.agentic.llm import LLMClient
from verifier.agentic.protocol import AgentResponse, parse_agent_response
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
            max_tokens=self.max_tokens,
        )
        return parse_agent_response(text)

    def _build_system_prompt(self) -> str:
        parts = [
            self.instructions.strip(),
            "",
            "You must respond with exactly one JSON object and no surrounding prose.",
            "The JSON object schema is:",
            '{"message": "short explanation", "tool_calls": [{"tool": "tool_name", "args": {}}]}',
            "Use tool_calls to request local runtime actions. Do not claim a tool was run unless it appears in the ledger.",
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
    uncovered_open_claim_ids = [
        claim.id
        for claim in state.claims
        if _status_value(claim.status) == ClaimStatus.OPEN.value and not claim.evidence
    ]
    return cast(dict[str, JsonValue], {
        "open_claim_ids": open_claim_ids,
        "uncovered_open_claim_ids": uncovered_open_claim_ids,
        "all_open_claims_have_evidence": not uncovered_open_claim_ids,
    })


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"
