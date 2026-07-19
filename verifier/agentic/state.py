"""Structured state objects for agentic verification runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, cast

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class Role(str, Enum):
    DESCRIBER = "describer"
    SKEPTIC = "skeptic"
    EXPERIMENTER = "experimenter"
    JUDGE = "judge"
    ORCHESTRATOR = "orchestrator"
    TOOL = "tool"


class ClaimStatus(str, Enum):
    OPEN = "open"
    CONFIRMED = "confirmed"
    REBUTTED = "rebutted"
    INCONCLUSIVE = "inconclusive"


class ClaimScope(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class DescriptionTaskStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class EvidenceKind(str, Enum):
    RUNTIME_PROBE = "runtime_probe"
    SOURCE_INSPECTION = "source_inspection"
    ARTIFACT_READ = "artifact_read"
    TOOL_ERROR = "tool_error"
    AGENT_ANALYSIS = "agent_analysis"


def utc_now_iso() -> str:
    """Return a compact UTC timestamp suitable for persisted JSON."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class ToolCall:
    tool: str
    args: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return {"tool": self.tool, "args": self.args}


@dataclass(slots=True)
class Turn:
    role: Role | str
    round: int
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "role": _enum_value(self.role),
            "round": self.round,
            "text": self.text,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


@dataclass(slots=True)
class ToolEvent:
    id: str
    tool: str
    args: dict[str, JsonValue]
    status: ToolStatus | str
    output: dict[str, JsonValue]
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "tool": self.tool,
            "args": self.args,
            "status": _enum_value(self.status),
            "output": self.output,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ArtifactRef:
    kind: Literal["probe_code", "stdout", "stderr", "source_snapshot", "json_result", "other"]
    path: str
    description: str
    sha256: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        data: dict[str, JsonValue] = {
            "kind": self.kind,
            "path": self.path,
            "description": self.description,
        }
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        return data


@dataclass(slots=True)
class DescriptionModel:
    contract_model: list[str] = field(default_factory=list)
    kernel_model: list[str] = field(default_factory=list)
    risk_map: list[str] = field(default_factory=list)
    scope_notes: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "contract_model": list(self.contract_model),
            "kernel_model": list(self.kernel_model),
            "risk_map": list(self.risk_map),
            "scope_notes": list(self.scope_notes),
            "open_questions": list(self.open_questions),
        }


@dataclass(slots=True)
class DescriptionTask:
    id: str
    reason_kind: str
    question: str
    requested_by: Role | str
    status: DescriptionTaskStatus | str = DescriptionTaskStatus.OPEN
    related_claims: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    request_turn: int | None = None
    response_summary: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    resolved_at: str | None = None
    resolved_turn: int | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "reason_kind": self.reason_kind,
            "question": self.question,
            "requested_by": _enum_value(self.requested_by),
            "status": _enum_value(self.status),
            "related_claims": list(self.related_claims),
            "source_refs": list(self.source_refs),
            "request_turn": self.request_turn,
            "response_summary": self.response_summary,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolved_turn": self.resolved_turn,
        }


@dataclass(slots=True)
class DescriptionUpdate:
    id: str
    summary: str
    task_ids: list[str] = field(default_factory=list)
    contract_model: list[str] = field(default_factory=list)
    kernel_model: list[str] = field(default_factory=list)
    risk_map: list[str] = field(default_factory=list)
    scope_notes: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    impact_on_claims: list[str] = field(default_factory=list)
    created_by: Role | str = Role.DESCRIBER
    turn: int | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "summary": self.summary,
            "task_ids": list(self.task_ids),
            "contract_model": list(self.contract_model),
            "kernel_model": list(self.kernel_model),
            "risk_map": list(self.risk_map),
            "scope_notes": list(self.scope_notes),
            "open_questions": list(self.open_questions),
            "impact_on_claims": list(self.impact_on_claims),
            "created_by": _enum_value(self.created_by),
            "turn": self.turn,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class Evidence:
    id: str
    kind: EvidenceKind | str
    tool_event_id: str | None
    summary: str
    supports: ClaimStatus | str
    data: dict[str, JsonValue] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "kind": _enum_value(self.kind),
            "tool_event_id": self.tool_event_id,
            "summary": self.summary,
            "supports": _enum_value(self.supports),
            "data": self.data,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class Claim:
    id: str
    statement: str
    rationale: str
    status: ClaimStatus | str
    raised_by: Role | str
    scope: ClaimScope | str = ClaimScope.UNKNOWN
    scope_rationale: str = ""
    scope_evidence: list[dict[str, JsonValue]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], {
            "id": self.id,
            "statement": self.statement,
            "rationale": self.rationale,
            "status": _enum_value(self.status),
            "raised_by": _enum_value(self.raised_by),
            "scope": _enum_value(self.scope),
            "scope_rationale": self.scope_rationale,
            "scope_evidence": self.scope_evidence,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        })


@dataclass(slots=True)
class RunState:
    entry: str | None = None
    artifact: dict[str, JsonValue] | None = None
    skills: list[str] = field(default_factory=list)
    description_model: DescriptionModel = field(default_factory=DescriptionModel)
    description_tasks: list[DescriptionTask] = field(default_factory=list)
    description_updates: list[DescriptionUpdate] = field(default_factory=list)
    history: list[Turn] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    verdict: dict[str, JsonValue] | None = None
    convergence: dict[str, JsonValue] | None = None
    skeptic_review: dict[str, JsonValue] | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], {
            "entry": self.entry,
            "artifact": self.artifact,
            "skills": self.skills,
            "description_model": self.description_model.to_dict(),
            "description_tasks": [task.to_dict() for task in self.description_tasks],
            "description_updates": [update.to_dict() for update in self.description_updates],
            "history": [turn.to_dict() for turn in self.history],
            "tool_events": [event.to_dict() for event in self.tool_events],
            "claims": [claim.to_dict() for claim in self.claims],
            "verdict": self.verdict,
            "convergence": self.convergence,
            "skeptic_review": self.skeptic_review,
        })


def _enum_value(value: Enum | str) -> str:
    return value.value if isinstance(value, Enum) else value
