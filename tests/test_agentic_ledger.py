from __future__ import annotations

import pytest

from verifier.agentic.ledger import ClaimLedger, LedgerError
from verifier.agentic.state import ClaimStatus, EvidenceKind, RunState


def test_record_claim_assigns_stable_ids() -> None:
    state = RunState()
    ledger = ClaimLedger(state)

    first = ledger.record_claim(statement="stride is ignored", rationale="source uses linear offsets")
    second = ledger.record_claim(statement="tail elements are skipped", rationale="mask excludes boundary")

    assert first.id == "c1"
    assert first.status == ClaimStatus.OPEN
    assert second.id == "c2"
    assert [claim["id"] for claim in ledger.read_claims()] == ["c1", "c2"]


def test_append_evidence_and_update_status() -> None:
    state = RunState()
    ledger = ClaimLedger(state)
    claim = ledger.record_claim(statement="stride is ignored", rationale="source uses linear offsets")

    evidence = ledger.append_evidence(
        claim_id=claim.id,
        kind=EvidenceKind.SOURCE_INSPECTION,
        tool_event_id="t1",
        summary="Line 12 indexes x + offsets without applying stride.",
        supports=ClaimStatus.CONFIRMED,
        data={"path": "kernel.py", "line": 12},
    )
    updated = ledger.update_claim_status(claim_id=claim.id, status=ClaimStatus.CONFIRMED)

    assert evidence.id == "c1.e1"
    assert updated.status == ClaimStatus.CONFIRMED
    assert updated.evidence[0].summary.startswith("Line 12")


def test_rejects_direct_confirmed_to_rebutted_transition() -> None:
    state = RunState()
    ledger = ClaimLedger(state)
    claim = ledger.record_claim(statement="stride is ignored", rationale="source uses linear offsets")
    ledger.append_evidence(
        claim_id=claim.id,
        kind=EvidenceKind.SOURCE_INSPECTION,
        tool_event_id="t1",
        summary="The source supports the original claim.",
        supports=ClaimStatus.CONFIRMED,
        data={"path": "kernel.py"},
    )
    ledger.update_claim_status(claim_id=claim.id, status=ClaimStatus.CONFIRMED)

    with pytest.raises(LedgerError, match="invalid claim status transition"):
        ledger.update_claim_status(claim_id=claim.id, status=ClaimStatus.REBUTTED)


def _ctx():
    from verifier.agentic.state import RunState, Role
    from verifier.agentic.tools.registry import ToolContext, build_core_registry

    state = RunState()
    return build_core_registry(), ToolContext(
        state=state, current_role=Role.SKEPTIC.value, current_turn=1
    ), state


_LONG = (
    "route_top1() breaks ties in favour of the highest expert index instead of the "
    "lowest index the contract requires, so a row whose top two logits are exactly "
    "equal is routed to the wrong expert."
)


def test_record_claim_refuses_a_restatement_of_an_existing_claim() -> None:
    """A rejected record_claim must not be recoverable by recording it twice.

    The usual rejection is an in_scope claim missing its scope_rationale, and
    the tempting fix is to send the claim again with the field filled in --
    which leaves the first copy in the ledger. Measured runs did this four
    times in eighteen claims, twice word for word, and every duplicate had to
    be probed and resolved again before a verdict was allowed, out of the same
    turn budget the run needs for real work.
    """
    registry, context, state = _ctx()
    first = registry.call("record_claim", {"statement": _LONG, "rationale": "r"}, context=context)
    assert first.get("error") is None

    again = registry.call("record_claim", {"statement": _LONG, "rationale": "r"}, context=context)
    assert again["error"]["type"] == "LedgerError"
    assert "restates claim c1" in again["error"]["message"]
    assert len(state.claims) == 1

    other = registry.call(
        "record_claim",
        {"statement": "The kernel omits the K-dimension mask on its final block, so loads "
                      "run past the end of the packed weight tensor whenever K is not a "
                      "multiple of BLOCK_SIZE_K.",
         "rationale": "r"},
        context=context,
    )
    assert other.get("error") is None, "a genuinely different claim must still be accepted"
    assert len(state.claims) == 2


def test_short_statements_are_not_treated_as_duplicates() -> None:
    """The duplicate check must not fire on short strings.

    "claim 0" and "claim 1" are a 90% textual match and share no meaning; only
    full propositions are compared.
    """
    registry, context, state = _ctx()
    for i in range(3):
        result = registry.call(
            "record_claim", {"statement": f"claim {i}", "rationale": "r"}, context=context
        )
        assert result.get("error") is None
    assert len(state.claims) == 3
