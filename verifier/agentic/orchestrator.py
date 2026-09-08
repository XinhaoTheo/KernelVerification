"""Minimal orchestrator for agentic tool-call execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence, cast

from .persistence import PersistedRun, persist_run
from .protocol import AgentResponse
from .state import ClaimStatus, DescriptionTaskStatus, JsonValue, Role, RunState, ToolCall, Turn
from .tools.registry import ToolContext, ToolRegistry, build_core_registry


class StopReason(str, Enum):
    """Why a run_* loop returned. Values are the stable strings persisted to run.json/transcripts."""

    VERDICT_RECORDED = "verdict_recorded"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    CLAIM_COVERAGE_REQUIRED = "claim_coverage_required"
    CLAIM_COVERAGE_STALLED = "claim_coverage_stalled"
    PROBE_OUTPUT_UNCONSUMED = "probe_output_unconsumed"
    SKEPTIC_REVIEW_REQUIRED = "skeptic_review_required"
    MORE_DEBATE_REQUESTED = "more_debate_requested"
    NO_OPEN_CLAIMS = "no_open_claims"
    MAX_ROUNDS_EXHAUSTED = "max_rounds_exhausted"

    def __str__(self) -> str:  # keep f-strings/prints as the plain value, not "StopReason.X"
        return self.value


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
    stop_reason: StopReason | str


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
                duration_s=response.duration_s,
                usage=response.usage,
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
        response = agent.act(state=self.state, tools=self.registry.list_tools(role=agent.role))
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

    def _run_agent_and_check_verdict(
        self,
        agent: Agent,
        *,
        outputs: list[dict],
        describer: Agent | None,
        stop_on_verdict: bool,
    ) -> StopReason | None:
        """Run one agent turn, then drain any description tasks it triggered.

        Extends `outputs` in place. Returns a StopReason if the caller should stop
        immediately (a verdict was just recorded), or None to keep looping.
        """
        outputs.extend(self.run_agent_once(agent))
        if stop_on_verdict and self.state.verdict is not None:
            return StopReason.VERDICT_RECORDED
        if _role_value(agent.role) != Role.DESCRIBER.value:
            outputs.extend(self.run_pending_description_tasks(describer))
            if stop_on_verdict and self.state.verdict is not None:
                return StopReason.VERDICT_RECORDED
        return None

    def run_agents_sequential(
        self,
        agents: Sequence[Agent],
        *,
        max_rounds: int,
        tool_budget: int | None = None,
        stop_on_verdict: bool = True,
        stop_when_no_open_claims: bool = False,
        require_claim_coverage: bool = False,
        force_verdict_on_last_round: bool = False,
    ) -> LoopResult:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        outputs: list[dict] = []
        start_tool_events = len(self.state.tool_events)

        skipped_judge_for_coverage = False
        describer = _first_agent_with_role(agents, Role.DESCRIBER)

        for round_no in range(1, max_rounds + 1):
            # The debate workflow tells the Judge when the budget is spent; a
            # sequential run had no equivalent, so an agent with tools could keep
            # opening claims until the rounds ran out and never decide. Reuse the
            # workflow's own close-out so the two arms differ in structure, not in
            # whether anything ever asks for an answer.
            final_round = force_verdict_on_last_round and round_no == max_rounds
            if final_round:
                notice = self._close_out_for_forced_verdict()
                forced_context: dict[str, JsonValue] = {
                    "unresolved_claims": notice["unresolved_claims"],
                    "skeptic_signed_off": notice["skeptic_signed_off"],
                }
                # A single turn is not enough to guarantee an answer: the agent
                # can spend it on one more probe and end the run with nothing.
                # The Judge has the same problem and solves it by keeping the
                # floor until it decides; give the last round the same loop.
                for agent in agents:
                    stop_reason = self._run_judge_until_decision(
                        agent, outputs=outputs, describer=describer,
                        stop_on_verdict=stop_on_verdict,
                    )
                    if self.state.verdict is not None:
                        # Same reason the workflow stamps this: a verdict reached
                        # on a spent budget with claims still open is not the same
                        # result as one reached after everything was settled, and
                        # record_verdict clears `convergence`, so it would
                        # otherwise leave no trace in the persisted run.
                        self.state.verdict["forced_final_round"] = cast(JsonValue, forced_context)
                    if stop_reason is not None:
                        return LoopResult(outputs, round_no, stop_reason)
                continue
            for agent in agents:
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, round_no - 1, StopReason.TOOL_BUDGET_EXHAUSTED)
                if (
                    require_claim_coverage
                    and _role_value(agent.role) == Role.JUDGE.value
                    and self.open_claim_ids()
                ):
                    skipped_judge_for_coverage = True
                    continue
                stop_reason = self._run_agent_and_check_verdict(
                    agent, outputs=outputs, describer=describer, stop_on_verdict=stop_on_verdict,
                )
                if stop_reason is not None:
                    return LoopResult(outputs, round_no, stop_reason)
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, round_no, StopReason.TOOL_BUDGET_EXHAUSTED)
            if stop_when_no_open_claims and self.state.claims and not self.has_open_claims():
                return LoopResult(outputs, round_no, StopReason.NO_OPEN_CLAIMS)

        if skipped_judge_for_coverage:
            return LoopResult(outputs, max_rounds, StopReason.CLAIM_COVERAGE_REQUIRED)
        return LoopResult(outputs, max_rounds, StopReason.MAX_ROUNDS_EXHAUSTED)

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
                    return LoopResult(outputs, debate_round - 1, StopReason.TOOL_BUDGET_EXHAUSTED)
                stop_reason = self._run_agent_and_check_verdict(
                    agent, outputs=outputs, describer=describer, stop_on_verdict=stop_on_verdict,
                )
                if stop_reason is not None:
                    return LoopResult(outputs, debate_round, stop_reason)
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round, StopReason.TOOL_BUDGET_EXHAUSTED)

            claim_rounds = 0
            claim_round_budget = self._claim_round_budget(
                max_claim_rounds=max_claim_rounds,
                max_claim_rounds_per_claim=max_claim_rounds_per_claim,
            )
            # On the last round a stalled coverage loop must not end the run with
            # no verdict: stop probing and fall through to the Judge, which is
            # then told what was left unresolved.
            can_force_verdict = judge is not None and debate_round >= max_debate_rounds

            def _coverage_stop(reason: StopReason) -> LoopResult | None:
                return None if can_force_verdict else LoopResult(outputs, debate_round, reason)

            while require_claim_coverage and self.open_claim_ids():
                if experimenter is None:
                    result = _coverage_stop(StopReason.CLAIM_COVERAGE_REQUIRED)
                    if result is not None:
                        return result
                    break
                force_probe_consumption = claim_rounds >= claim_round_budget and self.has_unconsumed_probe_events()
                if claim_rounds >= claim_round_budget and not force_probe_consumption:
                    result = _coverage_stop(StopReason.CLAIM_COVERAGE_REQUIRED)
                    if result is not None:
                        return result
                    break
                # A blown tool budget is a hard resource limit, not a debate
                # outcome, so it still stops the run outright.
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round, StopReason.TOOL_BUDGET_EXHAUSTED)

                before_claims = self._claim_progress_signature()
                before_description = self._description_progress_signature()
                before_tool_events = len(self.state.tool_events)
                stop_reason = self._run_agent_and_check_verdict(
                    experimenter, outputs=outputs, describer=describer, stop_on_verdict=stop_on_verdict,
                )
                claim_rounds += 1

                if stop_reason is not None:
                    return LoopResult(outputs, debate_round, stop_reason)
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round, StopReason.TOOL_BUDGET_EXHAUSTED)
                claims_changed = self._claim_progress_signature() != before_claims
                description_changed = self._description_progress_signature() != before_description
                probe_output_added = self._has_new_probe_event_since(before_tool_events)
                if force_probe_consumption and not claims_changed:
                    result = _coverage_stop(StopReason.PROBE_OUTPUT_UNCONSUMED)
                    if result is not None:
                        return result
                    break
                if not claims_changed and not probe_output_added and not description_changed:
                    result = _coverage_stop(StopReason.CLAIM_COVERAGE_STALLED)
                    if result is not None:
                        return result
                    break

            if judge is not None:
                # On the last round the run must end with a verdict rather than
                # a stop reason, so the gates below stop blocking and whatever is
                # still unresolved is disclosed to the Judge instead.
                final_round = debate_round >= max_debate_rounds

                if require_claim_coverage and self.open_claim_ids() and not final_round:
                    continue
                if debate_round < min_debate_rounds_before_judge:
                    self._record_internal_more_debate_request(
                        reason="minimum debate rounds before Judge not reached",
                    )
                    continue

                # The Skeptic's other speaking slot is at the top of the round,
                # before the Experimenter runs, so any probe run afterwards
                # invalidates a sign-off made there. Without a turn here the gate
                # below could never open in a round where the Experimenter did
                # any work, and the Judge would never be reached. Run the Skeptic
                # now, once the evidence it has to review actually exists. If it
                # raises new claims instead of signing off, the gate below sends
                # the run into another round, which is the intended outcome.
                if skeptic is not None and not self._skeptic_review_current():
                    # Mark the turn as a review so the Skeptic switches out of
                    # hypothesis-generation mode. Without this it arrives with
                    # its usual "raise concrete bug hypotheses" instructions and
                    # tends to open a fresh claim here, which sends the run into
                    # another round instead of letting the Judge rule.
                    self._record_skeptic_final_review_request()
                    stop_reason = self._run_agent_and_check_verdict(
                        skeptic, outputs=outputs, describer=describer, stop_on_verdict=stop_on_verdict,
                    )
                    if stop_reason is not None:
                        return LoopResult(outputs, debate_round, stop_reason)
                    if self._tool_budget_exhausted(start_tool_events, tool_budget):
                        return LoopResult(outputs, debate_round, StopReason.TOOL_BUDGET_EXHAUSTED)
                    if require_claim_coverage and self.open_claim_ids() and not final_round:
                        continue

                if skeptic is not None and not self._skeptic_review_current() and not final_round:
                    self._record_internal_more_debate_request(
                        reason="Skeptic must review the latest evidence and call record_no_new_claims before Judge",
                    )
                    continue

                forced_context: dict[str, JsonValue] | None = None
                if final_round:
                    # Discloses anything still open (and whether the Skeptic ever
                    # signed off) to the Judge, and settles the ledger enough for
                    # record_verdict to accept any of its three verdicts.
                    notice = self._close_out_for_forced_verdict()
                    forced_context = {
                        "unresolved_claims": notice["unresolved_claims"],
                        "skeptic_signed_off": notice["skeptic_signed_off"],
                    }
                else:
                    self.state.convergence = None
                stop_reason = self._run_judge_until_decision(
                    judge, outputs=outputs, describer=describer, stop_on_verdict=stop_on_verdict,
                )
                if forced_context is not None and self.state.verdict is not None:
                    # record_verdict clears `convergence`, so without this the
                    # persisted run would not show that this verdict was reached
                    # under a spent budget. A "trust" returned alongside three
                    # unresolved claims is not the same result as one returned
                    # after a clean sign-off, and a reader has to be able to tell.
                    self.state.verdict["forced_final_round"] = cast(JsonValue, forced_context)
                if stop_reason is not None:
                    return LoopResult(outputs, debate_round, stop_reason)
                if self._tool_budget_exhausted(start_tool_events, tool_budget):
                    return LoopResult(outputs, debate_round, StopReason.TOOL_BUDGET_EXHAUSTED)
                if self._more_debate_requested():
                    if debate_round < max_debate_rounds:
                        continue
                    return LoopResult(outputs, debate_round, StopReason.MORE_DEBATE_REQUESTED)
            elif stop_when_no_open_claims and self.state.claims and not self.has_open_claims():
                return LoopResult(outputs, debate_round, StopReason.NO_OPEN_CLAIMS)

        if require_claim_coverage and self.open_claim_ids():
            return LoopResult(outputs, max_debate_rounds, StopReason.CLAIM_COVERAGE_REQUIRED)
        return LoopResult(outputs, max_debate_rounds, StopReason.MAX_ROUNDS_EXHAUSTED)

    def _run_judge_until_decision(
        self,
        judge: Agent,
        *,
        outputs: list[dict],
        describer: Agent | None,
        stop_on_verdict: bool,
        max_turns: int = 3,
    ) -> StopReason | None:
        """Let the Judge keep the floor until it actually decides something.

        Every other role gets to work until it is done: the Experimenter has the
        claim-coverage loop, the Describer is drained up to max_turns. The Judge
        had a single turn per round, so a turn spent on inspect_problem or
        read_artifact_file -- which its own instructions tell it to use when it
        needs context -- ended the round with nothing recorded. Give it several
        turns, stopping as soon as it records a verdict, asks for more debate, or
        stops making progress.
        """
        for _ in range(max_turns):
            before_tool_events = len(self.state.tool_events)
            stop_reason = self._run_agent_and_check_verdict(
                judge, outputs=outputs, describer=describer, stop_on_verdict=stop_on_verdict,
            )
            if stop_reason is not None:
                return stop_reason
            if self.state.verdict is not None or self._more_debate_requested():
                return None
            if len(self.state.tool_events) == before_tool_events:
                # A turn that called nothing will not call anything next time
                # either; spending more budget on it is waste.
                return None
        return None

    def _close_out_for_forced_verdict(self) -> dict[str, JsonValue]:
        """Hand an exhausted run to the Judge instead of ending with no verdict.

        On the last debate round the run must produce a verdict, so anything the
        debate never settled is disclosed rather than hidden: each still-open
        claim gets an explicit "never resolved" evidence entry and moves to
        inconclusive (record_verdict refuses trust/reject while claims are open,
        and ClaimLedger refuses a status change with no supporting evidence), and
        the returned notice tells the Judge exactly what was left hanging.
        """
        from .ledger import ClaimLedger
        from .state import utc_now_iso

        ledger = ClaimLedger(self.state)
        unresolved = self.open_claim_ids()
        for claim_id in unresolved:
            ledger.append_evidence(
                claim_id=claim_id,
                kind="agent_analysis",
                summary=(
                    "Debate budget was exhausted before this claim was resolved. It is recorded "
                    "as inconclusive so the Judge can weigh it as an open question, not because "
                    "any evidence settled it."
                ),
                supports=ClaimStatus.INCONCLUSIVE,
                data={"unresolved_at_budget_exhaustion": True},
            )
            ledger.update_claim_status(claim_id=claim_id, status=ClaimStatus.INCONCLUSIVE)

        notice: dict[str, JsonValue] = {
            "request": "final_verdict_required",
            "reason": (
                "This is the final debate round: a verdict must be recorded now. The claims "
                "listed in unresolved_claims were never settled by evidence, and "
                "skeptic_signed_off reports whether the Skeptic ever confirmed it had no "
                "further concerns. Weigh both when choosing the verdict and say so in the reason."
            ),
            "unresolved_claims": cast(JsonValue, unresolved),
            "skeptic_signed_off": self._skeptic_review_current(),
            "created_at": utc_now_iso(),
        }
        self.state.convergence = notice
        return notice

    def _record_skeptic_final_review_request(self) -> None:
        """Signal the Skeptic that this turn is a review, not a new attack round."""
        from .state import utc_now_iso

        self.state.convergence = {
            "request": "skeptic_final_review",
            "reason": (
                "This round's probes are finished and the Judge is waiting. Review the new "
                "evidence and call record_no_new_claims unless it exposes a material, testable, "
                "in-scope problem that existing claims do not already cover."
            ),
            "focus_claims": [],
            "created_at": utc_now_iso(),
        }

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

        # Tools that introduce new evidence about the kernel's behavior, and so
        # invalidate a review made before they ran.
        #
        # request_description / record_description_update are deliberately NOT
        # here. They clarify the contract rather than produce new evidence, and
        # the orchestrator drains pending description tasks automatically after
        # every non-Describer turn -- including right after the Skeptic's. With
        # them in this set, a Skeptic that asked a question and signed off in the
        # same turn had its own sign-off invalidated by the answer it asked for,
        # permanently blocking the Judge.
        stale_tools = {
            "record_claim",
            "append_evidence",
            "update_claim_status",
            "run_python_probe",
            "run_claim_probe",
            "finalize_probe_evidence",
        }
        return not any(event.tool in stale_tools for event in self.state.tool_events[reviewed_count:])

    def has_open_claims(self) -> bool:
        return any(_status_value(claim.status) == ClaimStatus.OPEN.value for claim in self.state.claims)

    def has_open_description_tasks(self) -> bool:
        return any(
            _status_value(task.status) == DescriptionTaskStatus.OPEN.value
            for task in self.state.description_tasks
        )

    def open_claim_ids(self) -> list[str]:
        return [
            claim.id
            for claim in self.state.claims
            if _status_value(claim.status) == ClaimStatus.OPEN.value
        ]

    def has_unconsumed_probe_events(self) -> bool:
        return bool(self._unconsumed_probe_event_ids())

    def _claim_round_budget(self, *, max_claim_rounds: int, max_claim_rounds_per_claim: int) -> int:
        open_claim_count = len(self.open_claim_ids())
        return max(max_claim_rounds, open_claim_count * max_claim_rounds_per_claim)

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
            if event.tool == "run_claim_probe"
            and _status_value(event.status) == "ok"
            and event.output.get("claim_id")
            and event.id not in consumed_event_ids
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
