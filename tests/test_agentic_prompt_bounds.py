"""Every string and list in the rendered run state must be bounded.

Per-field caps only bound the fields someone remembered to cap. A new state
field, a dict nested inside a tool output, or a list an agent can append to
without limit would otherwise reach the prompt at full length and be re-sent on
every call. These tests assert the property generically, so adding an unbounded
field fails here rather than showing up as a bill.
"""
from __future__ import annotations

import json

from verifier.agentic import state as S
from verifier.agentic.agents.base import _DEFAULT_LIST_LIMIT, _state_for_prompt

BIG = "X" * 40_000
HARD_CEILING = 12_100  # the artifact source budget, the largest allowed anywhere


def _oversized_state() -> S.RunState:
    """A state with every string field stuffed past every cap."""
    big3 = [BIG] * 3
    st = S.RunState()
    st.entry = "toy"
    st.artifact = {"kernel_code": BIG, "test_code": BIG, "problem_text": BIG, "notes": BIG}
    st.description_model = S.DescriptionModel(
        contract_model=big3, kernel_model=big3, risk_map=big3,
        scope_notes=big3, open_questions=big3,
    )
    st.description_tasks = [
        S.DescriptionTask(id="d1", reason_kind="k", question=BIG,
                          requested_by="skeptic", response_summary=BIG)
    ]
    st.description_updates = [
        S.DescriptionUpdate(id="u1", summary=BIG, contract_model=big3,
                            risk_map=big3, impact_on_claims=big3)
    ]
    st.history = [
        S.Turn(role="skeptic", round=1, text=BIG,
               tool_calls=[S.ToolCall(tool="run_claim_probe", args={"code": BIG})])
    ]
    st.tool_events = [
        S.ToolEvent(id="t1", tool="run_claim_probe", args={"code": BIG},
                    status="ok", output={"stdout": BIG}),
        # A tool whose payload nests the bulk one level down. The per-field cap
        # only looked at top-level strings, so this used to pass through whole.
        S.ToolEvent(id="t2", tool="retrieve_experiment_history", args={},
                    status="ok", output={"events": [{"output": {"stdout": BIG}}]}),
    ]
    st.claims = [
        S.Claim(id="c1", statement=BIG, rationale=BIG, status="open", raised_by="skeptic",
                scope_rationale=BIG, scope_evidence=[{"note": BIG}],
                evidence=[S.Evidence(id="e1", kind="runtime", tool_event_id="t1",
                                     summary=BIG, supports="confirmed",
                                     data={"raw": BIG},
                                     artifacts=[S.ArtifactRef(kind="stdout", path=BIG,
                                                              description=BIG)])])
    ]
    return st


def _strings(node, path=""):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings(value, f"{path}[{index}]")


def test_no_string_in_the_rendered_state_is_unbounded() -> None:
    rendered = _state_for_prompt(_oversized_state(), role="skeptic")
    oversized = {p: len(v) for p, v in _strings(rendered, "state") if len(v) > HARD_CEILING}
    assert not oversized, f"unbounded strings reached the prompt: {oversized}"


def test_nested_tool_payloads_are_bounded_too() -> None:
    # retrieve_experiment_history returns {"events": [...]}: a list, not a string,
    # so a top-level-only cap left the probe output underneath it at full length.
    rendered = _state_for_prompt(_oversized_state(), role="skeptic")
    event = next(e for e in rendered["tool_events"] if e["tool"] == "retrieve_experiment_history")
    assert len(json.dumps(event)) < HARD_CEILING


def test_judge_sees_the_full_ledger_but_still_bounded() -> None:
    # The Judge is exempt from claim compaction; it must not be exempt from bounds.
    rendered = _state_for_prompt(_oversized_state(), role="judge")
    oversized = {p: len(v) for p, v in _strings(rendered, "state") if len(v) > HARD_CEILING}
    assert not oversized, f"unbounded strings reached the Judge prompt: {oversized}"


def test_unbounded_lists_are_capped_with_a_visible_marker() -> None:
    st = _oversized_state()
    st.claims = st.claims * 200
    rendered = _state_for_prompt(st, role="skeptic")
    assert len(rendered["claims"]) == _DEFAULT_LIST_LIMIT + 1
    assert "more items" in str(rendered["claims"][-1])


def test_normal_sized_content_is_passed_through_untouched() -> None:
    st = S.RunState()
    st.entry = "toy"
    st.artifact = {"kernel_code": "def f(): pass", "problem_text": "add one"}
    st.claims = [S.Claim(id="c1", statement="stride handling is wrong",
                         rationale="the kernel assumes contiguous input",
                         status="open", raised_by="skeptic")]
    rendered = _state_for_prompt(st, role="skeptic")
    assert "truncated" not in json.dumps(rendered)
    assert rendered["claims"][0]["statement"] == "stride handling is wrong"


def test_evidence_fields_survive_at_their_own_budget() -> None:
    """The generic backstop must not claw back what a per-field rule kept.

    `_claims_for_prompt` decides how much of an evidence summary an open claim
    keeps; `_clamp` runs afterwards and would silently re-cut it to the generic
    default unless that path has its own budget.
    """
    summary = "S" * 3000
    st = S.RunState()
    st.claims = [
        S.Claim(id="c1", statement="s", rationale="r", status="open", raised_by="skeptic",
                evidence=[S.Evidence(id="e1", kind="runtime_probe", tool_event_id="t1",
                                     summary=summary, supports="open", data={})])
    ]
    rendered = _state_for_prompt(st, role="skeptic")
    assert rendered["claims"][0]["evidence"][0]["summary"] == summary


def test_inconclusive_claims_are_not_compacted_like_settled_ones() -> None:
    """An inconclusive claim still needs work, so the next turn must see the reasoning."""
    st = S.RunState()
    st.claims = [
        S.Claim(id="c1", statement="s", rationale="why this might be wrong",
                status="inconclusive", raised_by="skeptic", scope_rationale="domain note")
    ]
    claim = _state_for_prompt(st, role="skeptic")["claims"][0]
    assert claim["rationale"] == "why this might be wrong"
    assert claim["scope_rationale"] == "domain note"
