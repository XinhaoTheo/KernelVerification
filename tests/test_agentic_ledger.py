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
    ledger.update_claim_status(claim_id=claim.id, status=ClaimStatus.CONFIRMED)

    with pytest.raises(LedgerError, match="invalid claim status transition"):
        ledger.update_claim_status(claim_id=claim.id, status=ClaimStatus.REBUTTED)

