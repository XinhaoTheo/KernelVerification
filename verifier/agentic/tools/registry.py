"""Small typed tool registry for orchestrator-dispatched local tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from verifier.agentic.state import JsonValue, Role, RunState, ToolEvent, ToolStatus

ToolHandler = Callable[["ToolContext", dict[str, JsonValue]], dict[str, JsonValue]]

_AGENT_ROLES = frozenset(
    {
        Role.DESCRIBER.value,
        Role.SKEPTIC.value,
        Role.EXPERIMENTER.value,
        Role.JUDGE.value,
    }
)
_READ_ROLES = _AGENT_ROLES | {Role.ORCHESTRATOR.value}


class ToolRegistryError(ValueError):
    """Raised for duplicate, missing, or invalid tool calls."""


@dataclass(slots=True)
class ToolContext:
    state: RunState
    dataset_dir: Path | None = None
    run_dir: Path | None = None
    current_tool_event_id: str | None = None
    current_role: str | None = None
    current_turn: int | None = None
    current_turn_tool_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, JsonValue]
    allowed_roles: frozenset[str]
    handler: ToolHandler = field(repr=False)

    def public_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(slots=True)
class ToolRegistry:
    _tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, JsonValue],
        allowed_roles: frozenset[str] | set[str] | tuple[str, ...],
        handler: ToolHandler,
    ) -> None:
        if name in self._tools:
            raise ToolRegistryError(f"tool already registered: {name}")
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            allowed_roles=frozenset(_role_value(role) for role in allowed_roles),
            handler=handler,
        )

    def call(
        self,
        name: str,
        args: dict[str, JsonValue] | None,
        *,
        context: ToolContext,
    ) -> dict[str, JsonValue]:
        clean_args = args or {}
        event_id = self._next_tool_event_id(context)
        previous_event_id = context.current_tool_event_id
        context.current_tool_event_id = event_id
        context.current_turn_tool_counts[name] = context.current_turn_tool_counts.get(name, 0) + 1

        try:
            if name not in self._tools:
                raise ToolRegistryError(f"unknown tool: {name}")
            spec = self._tools[name]
            self._authorize(spec, context)
            self._validate_args(name, clean_args)
            output = spec.handler(context, clean_args)
        except Exception as exc:
            output = _error_output(event_id, exc)
            context.state.tool_events.append(
                ToolEvent(id=event_id, tool=name, args=clean_args, status=ToolStatus.ERROR, output=output)
            )
            return output
        finally:
            context.current_tool_event_id = previous_event_id

        context.state.tool_events.append(
            ToolEvent(
                id=event_id,
                tool=name,
                args=clean_args,
                status=ToolStatus.OK,
                output=output,
            )
        )
        return output

    def list_tools(self, *, role: Role | str | None = None) -> list[dict[str, JsonValue]]:
        if role is None:
            return [spec.public_dict() for spec in self._tools.values()]
        role_value = _role_value(role)
        return [
            spec.public_dict()
            for spec in self._tools.values()
            if role_value in spec.allowed_roles
        ]

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolRegistryError(f"unknown tool: {name}")
        return self._tools[name]

    def _validate_args(self, name: str, args: dict[str, JsonValue]) -> None:
        schema = self._tools[name].input_schema
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise ToolRegistryError(f"tool {name} has invalid schema: required must be a list")
        for key in required:
            if key not in args:
                raise ToolRegistryError(f"tool {name} missing required arg: {key}")

        if schema.get("additionalProperties") is False:
            properties = schema.get("properties", {})
            if not isinstance(properties, dict):
                raise ToolRegistryError(f"tool {name} has invalid schema: properties must be an object")
            extra = sorted(set(args) - set(properties))
            if extra:
                raise ToolRegistryError(f"tool {name} got unexpected args: {', '.join(extra)}")

    @staticmethod
    def _authorize(spec: ToolSpec, context: ToolContext) -> None:
        role = context.current_role
        if role is None or role not in spec.allowed_roles:
            actual_role = role or "unknown"
            raise ToolRegistryError(f"{actual_role} is not allowed to call {spec.name}")

    @staticmethod
    def _next_tool_event_id(context: ToolContext) -> str:
        return f"t{len(context.state.tool_events) + 1}"


def build_core_registry() -> ToolRegistry:
    from .artifacts import (
        inspect_kernel_source,
        inspect_kernel_source_schema,
        inspect_problem,
        inspect_problem_schema,
        list_artifact_files,
        list_artifact_files_schema,
        load_artifact,
        load_artifact_schema,
        read_artifact_file,
        read_artifact_file_schema,
    )
    from .claims import (
        append_evidence,
        append_evidence_schema,
        read_claim_ledger,
        read_claim_ledger_schema,
        record_claim,
        record_claim_schema,
        record_no_new_claims,
        record_no_new_claims_schema,
        update_claim_status,
        update_claim_status_schema,
    )
    from .description import (
        record_description_update,
        record_description_update_schema,
        request_description,
        request_description_schema,
    )
    from .execution import (
        finalize_probe_evidence,
        finalize_probe_evidence_schema,
        run_claim_probe,
        run_claim_probe_schema,
        run_python_probe,
        run_python_probe_schema,
    )
    from .history import retrieve_experiment_history, retrieve_experiment_history_schema
    from .verdict import (
        record_verdict,
        record_verdict_schema,
        request_more_debate,
        request_more_debate_schema,
    )

    registry = ToolRegistry()
    registry.register(
        name="load_artifact",
        description="Load a dataset artifact by entry name without making a verdict.",
        input_schema=load_artifact_schema(),
        allowed_roles={Role.ORCHESTRATOR.value},
        handler=load_artifact,
    )
    registry.register(
        name="inspect_kernel_source",
        description="Read a line-numbered kernel.py source slice from a dataset artifact.",
        input_schema=inspect_kernel_source_schema(),
        allowed_roles=_READ_ROLES,
        handler=inspect_kernel_source,
    )
    registry.register(
        name="inspect_problem",
        description="Read the problem/spec text for a dataset artifact.",
        input_schema=inspect_problem_schema(),
        allowed_roles=_READ_ROLES,
        handler=inspect_problem,
    )
    registry.register(
        name="list_artifact_files",
        description="List files inside one dataset artifact directory.",
        input_schema=list_artifact_files_schema(),
        allowed_roles=_READ_ROLES,
        handler=list_artifact_files,
    )
    registry.register(
        name="read_artifact_file",
        description="Read a controlled text file path inside one dataset artifact directory.",
        input_schema=read_artifact_file_schema(),
        allowed_roles=_READ_ROLES,
        handler=read_artifact_file,
    )
    registry.register(
        name="request_description",
        description="Request a Describer clarification for source, contract, probe, or verdict ambiguity.",
        input_schema=request_description_schema(),
        allowed_roles={Role.SKEPTIC.value, Role.EXPERIMENTER.value, Role.JUDGE.value},
        handler=request_description,
    )
    registry.register(
        name="record_description_update",
        description="Record a structured Describer update to the shared description model.",
        input_schema=record_description_update_schema(),
        allowed_roles={Role.DESCRIBER.value},
        handler=record_description_update,
    )
    registry.register(
        name="record_claim",
        description="Record one concrete, testable claim in the claim ledger.",
        input_schema=record_claim_schema(),
        allowed_roles={Role.SKEPTIC.value},
        handler=record_claim,
    )
    registry.register(
        name="read_claim_ledger",
        description="Read the current claim ledger.",
        input_schema=read_claim_ledger_schema(),
        allowed_roles=_READ_ROLES,
        handler=read_claim_ledger,
    )
    registry.register(
        name="record_no_new_claims",
        description="Record that the Skeptic reviewed the current evidence and found no new in-scope claims.",
        input_schema=record_no_new_claims_schema(),
        allowed_roles={Role.SKEPTIC.value},
        handler=record_no_new_claims,
    )
    registry.register(
        name="append_evidence",
        description="Attach source or runtime evidence to a claim.",
        input_schema=append_evidence_schema(),
        allowed_roles={Role.EXPERIMENTER.value},
        handler=append_evidence,
    )
    registry.register(
        name="update_claim_status",
        description="Update a claim status after evidence has been recorded.",
        input_schema=update_claim_status_schema(),
        allowed_roles={Role.EXPERIMENTER.value},
        handler=update_claim_status,
    )
    registry.register(
        name="run_python_probe",
        description="Run agent-generated Python probe code locally and return captured artifacts.",
        input_schema=run_python_probe_schema(),
        allowed_roles={Role.EXPERIMENTER.value},
        handler=run_python_probe,
    )
    registry.register(
        name="run_claim_probe",
        description="Run a Python probe for one claim and return an evidence draft bound to that claim.",
        input_schema=run_claim_probe_schema(),
        allowed_roles={Role.EXPERIMENTER.value},
        handler=run_claim_probe,
    )
    registry.register(
        name="finalize_probe_evidence",
        description="Interpret a run_claim_probe result, append runtime evidence, and update the claim status.",
        input_schema=finalize_probe_evidence_schema(),
        allowed_roles={Role.EXPERIMENTER.value},
        handler=finalize_probe_evidence,
    )
    registry.register(
        name="retrieve_experiment_history",
        description="Read prior run_python_probe tool events from the current run directory.",
        input_schema=retrieve_experiment_history_schema(),
        allowed_roles={Role.SKEPTIC.value, Role.EXPERIMENTER.value, Role.JUDGE.value},
        handler=retrieve_experiment_history,
    )
    registry.register(
        name="request_more_debate",
        description="Record that the Judge wants another debate round before final verdict.",
        input_schema=request_more_debate_schema(),
        allowed_roles={Role.JUDGE.value},
        handler=request_more_debate,
    )
    registry.register(
        name="record_verdict",
        description="Record the Judge agent's final evidence-based verdict.",
        input_schema=record_verdict_schema(),
        allowed_roles={Role.JUDGE.value},
        handler=record_verdict,
    )
    return registry


def _role_value(role: Role | str) -> str:
    return role.value if isinstance(role, Role) else str(role)


def _error_output(event_id: str, exc: Exception) -> dict[str, JsonValue]:
    return {
        "ok": False,
        "event_id": event_id,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "recoverable": True,
        },
    }
