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
            user=self._build_user_prompt(state=state),
            tools=tools,
            max_tokens=self.max_tokens,
        )
        metrics = getattr(self.llm_client, "last_metrics", None)
        try:
            response = parse_agent_response(text)
        except ProtocolError as exc:
            raise ProtocolError(f"{exc} | raw_response={_snippet(text)!r}") from exc
        if metrics is not None:
            response.duration_s = metrics.duration_s
            response.usage = metrics.to_dict()
        return response

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

    def _build_user_prompt(self, *, state: RunState) -> str:
        """The run state only.

        The tool definitions are NOT repeated here: they already reach the model
        through the provider's native tool-calling parameter (see
        `_anthropic_tool_specs` / `_openai_*_tool_specs` in llm.py), which is the
        copy the provider validates tool_call arguments against. Rendering the
        same schemas a second time as prompt text cost 3.6k-7.5k characters on
        every call -- 11% of the Experimenter's prompt -- and told the model
        nothing it could not already see.
        """
        return "\n".join(
            [
                "=== Current Run State ===",
                json.dumps(
                    _state_for_prompt(state, role=_role_value(self.role)),
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                "",
                # An agent that cannot see what it already has will spend a turn
                # fetching it. One did exactly that: it read a 68-byte meta.json
                # and re-read a range of the kernel already in its prompt, which
                # left the claim ledger empty for the round and skipped the
                # Experimenter entirely.
                "artifact.kernel_code above is the complete kernel source with line "
                "numbers, and tool_events already carries problem.txt and the "
                "artifact's file list. Do not spend a turn re-reading them. Read "
                "what you have and act on it.",
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


def _turn_for_prompt(turn) -> dict[str, JsonValue]:
    """Turn.to_dict() minus timing/token-usage bookkeeping agents have no use for."""
    data = turn.to_dict()
    data.pop("duration_s", None)
    data.pop("usage", None)
    return data


# Fields that carry evidence get a budget set so normal content never reaches it:
# a verdict turning on a line that a cost-saving cap deleted is indistinguishable
# from a verdict the system got wrong on its own, which would make the benchmark
# measure these constants instead of the debate. Cost is handled by the free
# measures (native tool schemas sent once, prompt cache) and revisited once a
# real run has produced real numbers.
_EVIDENCE_SUMMARY_LIMIT_OPEN = 4000       # still being argued: keep the reasoning
_EVIDENCE_SUMMARY_LIMIT_RESOLVED = 800    # confirmed/rebutted: the conclusion is enough
_PROBE_SOURCE_LIMIT = 4000                # a real Triton probe runs 1.5k-3k chars
_PROBE_OUTPUT_LIMIT = 4000                # sweep + conclusion; head+tail keeps both ends
_EVIDENCE_STDOUT_LIMIT = 8000             # the permanent verbatim copy in the ledger
_TURN_MESSAGE_LIMIT = 1500

# --- Generic size bound on the rendered state -------------------------------
#
# Every field above is a per-field cap, which only bounds the fields someone
# remembered to cap. Anything else -- a new state field, a dict nested inside a
# tool output, a list an agent can append to without limit -- reaches the prompt
# at full length. `_clamp` is the backstop: it walks the whole rendered state and
# bounds EVERY string and list, so a field is bounded by default and only
# oversized by explicit exception.
#
# Paths are dot-joined keys with list indices dropped, and a limit applies to a
# path and everything beneath it, so "tool_events.output" also bounds
# `tool_events[i].output.events[j].output.stdout`.
_DEFAULT_STRING_LIMIT = 2000
_DEFAULT_LIST_LIMIT = 50
_HISTORY_RETRIEVAL_LIMIT = 8000
# Probe output is written "sweep, then conclusion": the decisive line is the LAST
# one. Head-only truncation therefore drops exactly the part the probe was run
# for. These fields keep both ends instead.
_MIDDLE_TRUNCATE_KEYS = frozenset({"stdout", "stderr"})
_PATH_STRING_LIMITS: dict[str, int] = {
    # The artifact under test: the agents' primary reading material.
    "artifact.kernel_code": 12000,
    "history.text": _TURN_MESSAGE_LIMIT,
    "artifact.test_code": 12000,
    # Probe source and probe output, already capped by _tool_event_for_prompt;
    # repeated here so the same bound reaches nested tool payloads.
    "tool_events.args": _PROBE_SOURCE_LIMIT,
    "tool_events.output": _PROBE_OUTPUT_LIMIT,
    # retrieve_experiment_history is the deliberate "I need the full output"
    # path: an agent calls it precisely because the resident 1200-char copy was
    # not enough to check an evidence summary against. Holding it to the same
    # 1200 would make the tool pointless, so it gets its own, larger budget.
    "tool_events.output.events": _HISTORY_RETRIEVAL_LIMIT,
    # Where a probe puts its decisive numbers. Deliberately generous -- trimming
    # a measurement is worse than paying for it -- but no longer unbounded.
    "claims.evidence.data": _EVIDENCE_STDOUT_LIMIT,
    # _claims_for_prompt already applies the open/resolved distinction; this only
    # stops the generic default from clawing back what that decided to keep.
    "claims.evidence.summary": _EVIDENCE_SUMMARY_LIMIT_OPEN,
}


def _clamp(value: JsonValue, path: str = "") -> JsonValue:
    """Bound every string and list in `value`, recursively."""
    if isinstance(value, str):
        limit = _limit_for(path)
        if path.rsplit(".", 1)[-1] in _MIDDLE_TRUNCATE_KEYS:
            return _truncate_middle(value, limit)
        return _truncate(value, limit)
    if isinstance(value, dict):
        return cast(JsonValue, {
            key: _clamp(item, f"{path}.{key}" if path else str(key))
            for key, item in value.items()
        })
    if isinstance(value, list):
        clamped = [_clamp(item, path) for item in value[:_DEFAULT_LIST_LIMIT]]
        dropped = len(value) - _DEFAULT_LIST_LIMIT
        if dropped > 0:
            clamped.append(f"...[truncated, {dropped} more items]")
        return cast(JsonValue, clamped)
    return value


def _limit_for(path: str) -> int:
    """Longest matching path prefix wins, so a nested override beats its parent."""
    best = _DEFAULT_STRING_LIMIT
    best_len = -1
    for prefix, limit in _PATH_STRING_LIMITS.items():
        if (path == prefix or path.startswith(prefix + ".")) and len(prefix) > best_len:
            best, best_len = limit, len(prefix)
    return best


def _role_value(role) -> str:
    return role.value if isinstance(role, Role) else str(role)


def _claims_for_prompt(state: RunState, *, role: str | None) -> list[dict[str, JsonValue]]:
    """Render the claim ledger for one agent's prompt.

    `claims` is the largest state field that grows with run length: every claim
    carries every piece of evidence attached to it, and the whole list is re-sent
    on every LLM call, so a long run pays for the entire ledger many times over.

    The Judge writes the final verdict and runs once per round, so it still gets
    the full ledger. Working agents get resolved claims compacted to their
    conclusion, and evidence prose trimmed. Evidence `data` -- where a probe puts
    its decisive numbers -- keeps a far larger budget than prose (see
    `_PATH_STRING_LIMITS`), so trimming costs narrative long before it could cost
    a measurement.
    """
    if role == Role.JUDGE.value:
        return [claim.to_dict() for claim in state.claims]

    rendered: list[dict[str, JsonValue]] = []
    for claim in state.claims:
        data = claim.to_dict()
        # "inconclusive" means the claim still needs work, so it is compacted like
        # an open claim: the next Skeptic/Experimenter turn has to see how far the
        # last one got. Only confirmed and rebutted claims are actually settled.
        resolved = _status_value(claim.status) in {
            ClaimStatus.CONFIRMED.value,
            ClaimStatus.REBUTTED.value,
        }
        limit = _EVIDENCE_SUMMARY_LIMIT_RESOLVED if resolved else _EVIDENCE_SUMMARY_LIMIT_OPEN

        evidence = []
        for item in cast(list[dict[str, JsonValue]], data.get("evidence", [])):
            trimmed = dict(item)
            trimmed["summary"] = _truncate(str(trimmed.get("summary", "")), limit)
            trimmed.pop("artifacts", None)
            evidence.append(trimmed)
        data["evidence"] = cast(JsonValue, evidence)

        if resolved:
            # A settled claim only needs to say what it was and how it settled.
            for key in ("rationale", "scope_evidence", "scope_rationale"):
                data.pop(key, None)
        rendered.append(data)
    return rendered


def _number_lines(source: str) -> str:
    """Render source with line numbers, the way inspect_kernel_source does.

    Agents cite line numbers in claims and verdicts, and that is the only reason
    the preload used to call inspect_kernel_source on top of load_artifact --
    leaving two copies of the same file in the prompt (one raw, one numbered),
    both re-sent on every later turn. Numbering the copy that is already there
    removes the second one without taking away what it was for.
    """
    return "\n".join(f"{i}: {line}" for i, line in enumerate(source.splitlines(), start=1))


def _state_for_prompt_unbounded(state: RunState, *, role: str | None = None) -> dict[str, JsonValue]:
    artifact = dict(state.artifact or {})
    for key in ("kernel_code", "test_code"):
        if isinstance(artifact.get(key), str) and artifact[key]:
            artifact[key] = _truncate(_number_lines(str(artifact[key])), 12000)
    return cast(dict[str, JsonValue], {
        "entry": state.entry,
        "artifact": artifact,
        "history": [_turn_for_prompt(turn) for turn in state.history[-6:]],
        "description_model": state.description_model.to_dict(),
        "open_description_tasks": [
            task.to_dict()
            for task in state.description_tasks
            if _status_value(task.status) == "open"
        ],
        "recent_description_updates": [update.to_dict() for update in state.description_updates[-5:]],
        "tool_events": [event.to_dict() for event in state.tool_events[-12:]],
        "claims": _claims_for_prompt(state, role=role),
        "claim_coverage": _claim_coverage(state),
        "convergence": state.convergence,
        "skeptic_review": state.skeptic_review,
    })


def _state_for_prompt(state: RunState, *, role: str | None = None) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _clamp(cast(JsonValue, _state_for_prompt_unbounded(state, role=role))))


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
    """Keep the head, cut on a line boundary where one is available.

    Cutting mid-line can turn `max_rel_err=0.0034271` into `max_rel_err=0.003`:
    not information lost but information corrupted, since the short value still
    reads as a valid measurement. Backing up to the last newline avoids that
    whenever the text has line structure at all.
    """
    if len(text) <= limit:
        return text
    kept = _cut_at_line_boundary(text[:limit])
    return kept + f"\n...[truncated, {len(text) - len(kept)} more chars]"


def _truncate_middle(text: str, limit: int) -> str:
    """Keep both ends, dropping the middle, on line boundaries.

    For probe output the conclusion is the last line, so the tail is the half
    that must survive; the head is kept too because it usually states what the
    probe set up and measured. The tail gets the larger share.
    """
    if len(text) <= limit:
        return text
    head_budget = limit * 2 // 5
    tail_budget = limit - head_budget
    head = _cut_at_line_boundary(text[:head_budget])
    tail = text[len(text) - tail_budget:]
    newline = tail.find("\n")
    if newline != -1:
        tail = tail[newline + 1:]
    dropped = len(text) - len(head) - len(tail)
    return f"{head}\n...[truncated, {dropped} chars from the middle]\n{tail}"


def _cut_at_line_boundary(chunk: str) -> str:
    """Trim a hard cut back to the last complete line, if that keeps most of it."""
    newline = chunk.rfind("\n")
    # Only back up for text that actually has line structure near the cut; a
    # single long line has no better cut point than the hard one.
    if newline >= len(chunk) // 2:
        return chunk[:newline]
    return chunk


def _snippet(text: str, *, head: int = 1500, tail: int = 1500) -> str:
    """Head+tail snippet for debugging a malformed raw LLM response.

    Truncation-caused parse errors (e.g. "Unterminated string") are visible
    near the end; prose-wrapping errors ("Expecting value") are visible near
    the start. Keeping both is more useful here than a single-sided cut.
    """
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n...[{len(text) - head - tail} chars omitted]...\n{text[-tail:]}"
