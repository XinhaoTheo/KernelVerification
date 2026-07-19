"""Minimal orchestrator for agentic tool-call execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from .persistence import PersistedRun, persist_run
from .protocol import AgentResponse
from .state import ClaimStatus, DescriptionTaskStatus, Role, RunState, ToolCall, Turn
from .tools.registry import ToolContext, ToolRegistry, build_core_registry

_DEFAULT_SKILLS = [
    "kernel-verification.md",
    "evidence-driven-review.md",
    "claim-lifecycle.md",
    "experiment-design.md",
    "adversarial-precision.md",
    "metric-selection.md",
    "scope-policy.md",
    "convergence.md",
]


class Agent(Protocol):
    role: Role | str

    def act(self, *, state: RunState, tools: list[dict]) -> AgentResponse:
        """Return the agent's next JSON-protocol action."""
        ...


@dataclass(slots=True)
class LoopResult:
    outputs: list[dict]
    rounds_completed: int
    stop_reason: str


@dataclass(slots=True)
class AgenticOrchestrator:
    state: RunState = field(default_factory=RunState)
    registry: ToolRegistry = field(default_factory=build_core_registry)
    dataset_dir: Path | None = None
    run_dir: Path = Path("agentic_runs") / "adhoc"
    round_index: int = 0

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if self.state.skills == []:
            self.state.skills = list(_DEFAULT_SKILLS)
        if self.round_index == 0 and self.state.history:
            self.round_index = max(turn.round for turn in self.state.history)

    def apply_agent_response(self, *, role: Role | str, response: AgentResponse) -> list[dict]:
        self.round_index += 1
        current_turn = self.round_index
        self.state.history.append(
            Turn(
                role=role,
                round=current_turn,
                text=response.message,
                tool_calls=response.tool_calls,
            )
        )

        context = ToolContext(
            state=self.state,
            dataset_dir=self.dataset_dir,
            run_dir=self.run_dir,
            current_role=_role_value(role),
            current_turn=current_turn,
        )
        outputs = []
        for call in response.tool_calls:
            output = self.registry.call(call.tool, call.args, context=context)
            outputs.append({"tool": call.tool, "output": output})
        return outputs

    def run_agent_once(self, agent: Agent) -> list[dict]:
        response = agent.act(state=self.state, tools=self.registry.list_tools())
        return self.apply_agent_response(role=agent.role, response=response)

    def run_pending_description_tasks(self, describer: Agent | None, *, max_turns: int = 3) -> list[dict]:
        if describer is None or not self.has_open_description_tasks():
            return []
        outputs: list[dict] = []
        for _ in range(max_turns):
            if not self.has_open_description_tasks():
                break
            before = self._description_progress_signature()
            outputs.extend(self.run_agent_once(describer))
            if self._description_progress_signature() == before:
                break
        return outputs

    def run_agents_sequential(
        self,
        agents: Sequence[Agent],
        *,
        max_rounds: int,
        tool_budget: int | None = None,
        stop_on_verdict: bool = True,
        stop_when_no_open_claims: bool = False,
        require_claim_coverage: bool = False,
    ) -> LoopResult:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        outputs: list[dict] = []
        start_tool_events = len(self.state.tool_events)

        skipped_judge_for_coverage = False
        describer = _first_agent_with_role(agents, Role.DESCRIBER)

        for round_no in range(1, max_rounds + 1):
            for agent in agents:
                if tool_budget is not None and len(self.state.tool_events) - start_tool_events >= tool_budget:
                    return LoopResult(outputs, round_no - 1, "tool_budget_exhausted")
                if (
                    require_claim_coverage
                    and _role_value(agent.role) == Role.JUDGE.value
                    and self.has_uncovered_open_claims()
                ):
                    skipped_judge_for_coverage = True
                    continue
                outputs.extend(self.run_agent_once(agent))
                if stop_on_verdict and self.state.verdict is not None:
                    return LoopResult(outputs, round_no, "verdict_recorded")
                if _role_value(agent.role) != Role.DESCRIBER.value:
                    outputs.extend(self.run_pending_description_tasks(describer))
                    if stop_on_verdict and self.state.verdict is not None:
                        return LoopResult(outputs, round_no, "verdict_recorded")
                if tool_budget is not None and len(self.state.tool_events) - start_tool_events >= tool_budget:
                    return LoopResult(outputs, round_no, "tool_budget_exhausted")
            if stop_when_no_open_claims and self.state.claims and not self.has_open_claims():
                return LoopResult(outputs, round_no, "no_open_claims")

        if skipped_judge_for_coverage:
            return LoopResult(outputs, max_rounds, "claim_coverage_required")
        return LoopResult(outputs, max_rounds, "max_rounds_exhausted")

    def run_verification_workflow(
        self,
        agents: Sequence[Agent],
        *,
        max_debate_rounds: int,
        max_claim_rounds: int,
        max_claim_rounds_per_claim: int = 3,
        min_debate_rounds_before_judge: int = 1,
        tool_budget: int | None = None,
        stop_on_verdict: bool = True,
        stop_when_no_open_claims: bool = False,
        require_claim_coverage: bool = True,
    ) -> LoopResult:
        if max_debate_rounds < 1:
            raise ValueError("max_debate_rounds must be >= 1")
        if max_claim_rounds < 1:
            raise ValueError("max_claim_rounds must be >= 1")
        if max_claim_rounds_per_claim < 1:
            raise ValueError("max_claim_rounds_per_claim must be >= 1")
        if min_debate_rounds_before_judge < 1:
            raise ValueError("min_debate_rounds_before_judge must be >= 1")

        outputs: list[dict] = []
        start_tool_events = len(self.state.tool_events)
        debate_agents = [
            agent
            for agent in agents
            if _role_value(agent.role) not in {Role.EXPERIMENTER.value, Role.JUDGE.value}
        ]
        experimenter = _first_agent_with_role(agents, Role.EXPERIMENTER)
        judge = _first_agent_with_role(agents, Role.JUDGE)
        skeptic = _first_agent_with_role(agents, Role.SKEPTIC)
        describer = _first_agent_with_role(agents, Role.DESCRIBER)

        for debate_round in range(1, max_debate_rounds + 1):
            for agent in debate_agents:
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round - 1, "tool_budget_exhausted")
                outputs.extend(self.run_agent_once(agent))
                if stop_on_verdict and self.state.verdict is not None:
                    return LoopResult(outputs, debate_round, "verdict_recorded")
                if _role_value(agent.role) != Role.DESCRIBER.value:
                    outputs.extend(self.run_pending_description_tasks(describer))
                    if stop_on_verdict and self.state.verdict is not None:
                        return LoopResult(outputs, debate_round, "verdict_recorded")
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round, "tool_budget_exhausted")

            claim_rounds = 0
            claim_round_budget = self._claim_round_budget(
                max_claim_rounds=max_claim_rounds,
                max_claim_rounds_per_claim=max_claim_rounds_per_claim,
            )
            while require_claim_coverage and self.has_uncovered_open_claims():
                if experimenter is None:
                    return LoopResult(outputs, debate_round, "claim_coverage_required")
                force_probe_consumption = claim_rounds >= claim_round_budget and self.has_unconsumed_probe_events()
                if claim_rounds >= claim_round_budget and not force_probe_consumption:
                    return LoopResult(outputs, debate_round, "claim_coverage_required")
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round, "tool_budget_exhausted")

                before_claims = self._claim_progress_signature()
                before_description = self._description_progress_signature()
                before_tool_events = len(self.state.tool_events)
                outputs.extend(self.run_agent_once(experimenter))
                outputs.extend(self.run_pending_description_tasks(describer))
                claim_rounds += 1

                if stop_on_verdict and self.state.verdict is not None:
                    return LoopResult(outputs, debate_round, "verdict_recorded")
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round, "tool_budget_exhausted")
                claims_changed = self._claim_progress_signature() != before_claims
                description_changed = self._description_progress_signature() != before_description
                probe_output_added = self._has_new_probe_event_since(before_tool_events)
                if force_probe_consumption and not claims_changed:
                    return LoopResult(outputs, debate_round, "probe_output_unconsumed")
                if not claims_changed and not probe_output_added and not description_changed:
                    return LoopResult(outputs, debate_round, "claim_coverage_stalled")

            if judge is not None:
                if require_claim_coverage and self.has_uncovered_open_claims():
                    continue
                if debate_round < min_debate_rounds_before_judge:
                    self._record_internal_more_debate_request(
                        reason="minimum debate rounds before Judge not reached",
                    )
                    continue
                if skeptic is not None and not self._skeptic_review_current():
                    self._record_internal_more_debate_request(
                        reason="Skeptic must review the latest evidence and call record_no_new_claims before Judge",
                    )
                    if debate_round < max_debate_rounds:
                        continue
                    return LoopResult(outputs, debate_round, "skeptic_review_required")
                self.state.convergence = None
                outputs.extend(self.run_agent_once(judge))
                if stop_on_verdict and self.state.verdict is not None:
                    return LoopResult(outputs, debate_round, "verdict_recorded")
                outputs.extend(self.run_pending_description_tasks(describer))
                if stop_on_verdict and self.state.verdict is not None:
                    return LoopResult(outputs, debate_round, "verdict_recorded")
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round, "tool_budget_exhausted")
                if self._more_debate_requested():
                    if debate_round < max_debate_rounds:
                        continue
                    return LoopResult(outputs, debate_round, "more_debate_requested")
            elif stop_when_no_open_claims and self.state.claims and not self.has_open_claims():
                return LoopResult(outputs, debate_round, "no_open_claims")

        if require_claim_coverage and self.has_uncovered_open_claims():
            return LoopResult(outputs, max_debate_rounds, "claim_coverage_required")
        return LoopResult(outputs, max_debate_rounds, "max_rounds_exhausted")

    def _record_internal_more_debate_request(self, *, reason: str) -> None:
        from .state import utc_now_iso

        self.state.convergence = {
            "request": "more_debate",
            "reason": reason,
            "focus_claims": [],
            "created_at": utc_now_iso(),
        }

    def _more_debate_requested(self) -> bool:
        return bool(self.state.convergence and self.state.convergence.get("request") == "more_debate")

    def _skeptic_review_current(self) -> bool:
        review = self.state.skeptic_review
        if not review or review.get("decision") != "no_new_claims":
            return False
        reviewed_count = review.get("reviewed_tool_event_count")
        if not isinstance(reviewed_count, int) or reviewed_count < 0:
            return False

        stale_tools = {
            "record_claim",
            "append_evidence",
            "update_claim_status",
            "run_python_probe",
            "run_claim_probe",
            "finalize_probe_evidence",
            "request_description",
            "record_description_update",
        }
        return not any(event.tool in stale_tools for event in self.state.tool_events[reviewed_count:])

    def has_open_claims(self) -> bool:
        return any(_status_value(claim.status) == ClaimStatus.OPEN.value for claim in self.state.claims)

    def has_open_description_tasks(self) -> bool:
        return any(
            _status_value(task.status) == DescriptionTaskStatus.OPEN.value
            for task in self.state.description_tasks
        )

    def has_uncovered_open_claims(self) -> bool:
        return bool(self.uncovered_open_claim_ids())

    def uncovered_open_claim_ids(self) -> list[str]:
        return [
            claim.id
            for claim in self.state.claims
            if _status_value(claim.status) == ClaimStatus.OPEN.value and not claim.evidence
        ]

    def has_unconsumed_probe_events(self) -> bool:
        return bool(self._unconsumed_probe_event_ids())

    def _claim_round_budget(self, *, max_claim_rounds: int, max_claim_rounds_per_claim: int) -> int:
        uncovered_count = len(self.uncovered_open_claim_ids())
        return max(max_claim_rounds, uncovered_count * max_claim_rounds_per_claim)

    def _unconsumed_probe_event_ids(self) -> set[str]:
        consumed_event_ids = {
            evidence.tool_event_id
            for claim in self.state.claims
            for evidence in claim.evidence
            if evidence.tool_event_id
        }
        return {
            event.id
            for event in self.state.tool_events
            if event.tool in {"run_python_probe", "run_claim_probe"} and event.id not in consumed_event_ids
        }

    def _tool_budget_exhausted(self, start_tool_events: int, tool_budget: int | None) -> bool:
        return tool_budget is not None and len(self.state.tool_events) - start_tool_events >= tool_budget

    def _claim_progress_signature(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            (claim.id, _status_value(claim.status), len(claim.evidence))
            for claim in self.state.claims
        )

    def _description_progress_signature(self) -> tuple[tuple[tuple[str, str, str], ...], int, tuple[int, ...]]:
        return (
            tuple((task.id, _status_value(task.status), task.response_summary) for task in self.state.description_tasks),
            len(self.state.description_updates),
            (
                len(self.state.description_model.contract_model),
                len(self.state.description_model.kernel_model),
                len(self.state.description_model.risk_map),
                len(self.state.description_model.scope_notes),
                len(self.state.description_model.open_questions),
            ),
        )

    def _has_new_probe_event_since(self, start_index: int) -> bool:
        return any(event.tool in {"run_python_probe", "run_claim_probe"} for event in self.state.tool_events[start_index:])

    def persist(self, *, stop_reason: str | None = None) -> PersistedRun:
        return persist_run(self.state, self.run_dir, stop_reason=stop_reason)


def _first_agent_with_role(agents: Sequence[Agent], role: Role) -> Agent | None:
    for agent in agents:
        if _role_value(agent.role) == role.value:
            return agent
    return None


def _role_value(role) -> str:
    return role.value if hasattr(role, "value") else str(role)


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def build_context_response(entry: str) -> AgentResponse:
    """Build the deterministic context-loading action for an entry.

    This is not a verification battery. It only loads the artifact and basic
    source/spec context so agents can reason over concrete material.
    """
    return AgentResponse(
        message="Loading artifact context for the agent.",
        tool_calls=[
            ToolCall(tool="load_artifact", args={"entry": entry}),
            ToolCall(tool="inspect_problem", args={"entry": entry}),
            ToolCall(
                tool="inspect_kernel_source",
                args={"entry": entry, "start_line": 1, "end_line": 120},
            ),
        ],
    )
