"""Claim ledger operations for agentic verification."""

from __future__ import annotations

from dataclasses import dataclass

from .state import Claim, ClaimScope, ClaimStatus, Evidence, EvidenceKind, Role, RunState, utc_now_iso


class LedgerError(ValueError):
    """Raised when a claim ledger mutation is invalid."""


@dataclass(slots=True)
class ClaimLedger:
    state: RunState

    def record_claim(
        self,
        *,
        statement: str,
        rationale: str,
        raised_by: Role | str = Role.SKEPTIC,
        scope: ClaimScope | str = ClaimScope.UNKNOWN,
        scope_rationale: str = "",
        scope_evidence: list[dict] | None = None,
    ) -> Claim:
        statement = statement.strip()
        rationale = rationale.strip()
        scope_value = _normalize_claim_scope(scope)
        scope_rationale = scope_rationale.strip()
        scope_evidence = _normalize_scope_evidence(scope_evidence or [])
        _validate_scope_fields(scope_value, scope_rationale, scope_evidence)
        if not statement:
            raise LedgerError("claim statement must be non-empty")
        if not rationale:
            raise LedgerError("claim rationale must be non-empty")

        claim = Claim(
            id=self._next_claim_id(),
            statement=statement,
            rationale=rationale,
            status=ClaimStatus.OPEN,
            raised_by=raised_by,
            scope=scope_value,
            scope_rationale=scope_rationale,
            scope_evidence=scope_evidence,
        )
        self.state.claims.append(claim)
        return claim

    def append_evidence(
        self,
        *,
        claim_id: str,
        kind: EvidenceKind | str,
        summary: str,
        supports: ClaimStatus | str,
        tool_event_id: str | None = None,
        data: dict | None = None,
    ) -> Evidence:
        claim = self.get_claim(claim_id)
        summary = summary.strip()
        if not summary:
            raise LedgerError("evidence summary must be non-empty")
        support_status = _normalize_claim_status(supports)
        if support_status == ClaimStatus.OPEN:
            raise LedgerError("evidence cannot support open; use confirmed, rebutted, or inconclusive")

        evidence = Evidence(
            id=self._next_evidence_id(claim),
            kind=kind,
            tool_event_id=tool_event_id,
            summary=summary,
            supports=support_status,
            data=data or {},
        )
        claim.evidence.append(evidence)
        claim.updated_at = utc_now_iso()
        return evidence

    def update_claim_status(self, *, claim_id: str, status: ClaimStatus | str) -> Claim:
        claim = self.get_claim(claim_id)
        next_status = _normalize_claim_status(status)
        if claim.status == next_status:
            return claim
        self._validate_transition(_normalize_claim_status(claim.status), next_status)
        claim.status = next_status
        claim.updated_at = utc_now_iso()
        return claim

    def get_claim(self, claim_id: str) -> Claim:
        for claim in self.state.claims:
            if claim.id == claim_id:
                return claim
        raise LedgerError(f"unknown claim id: {claim_id}")

    def read_claims(self) -> list[dict]:
        return [claim.to_dict() for claim in self.state.claims]

    def _next_claim_id(self) -> str:
        return f"c{len(self.state.claims) + 1}"

    @staticmethod
    def _next_evidence_id(claim: Claim) -> str:
        return f"{claim.id}.e{len(claim.evidence) + 1}"

    @staticmethod
    def _validate_transition(current: ClaimStatus, next_status: ClaimStatus) -> None:
        if current == ClaimStatus.OPEN:
            return
        if next_status == ClaimStatus.INCONCLUSIVE:
            return
        if current == ClaimStatus.INCONCLUSIVE and next_status in {
            ClaimStatus.CONFIRMED,
            ClaimStatus.REBUTTED,
        }:
            return
        raise LedgerError(f"invalid claim status transition: {current.value} -> {next_status.value}")


def _normalize_claim_status(status: ClaimStatus | str) -> ClaimStatus:
    try:
        return status if isinstance(status, ClaimStatus) else ClaimStatus(status)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ClaimStatus)
        raise LedgerError(f"invalid claim status {status!r}; expected one of: {valid}") from exc



def _validate_scope_fields(scope: ClaimScope, scope_rationale: str, scope_evidence: list[dict]) -> None:
    if scope != ClaimScope.IN_SCOPE:
        return
    if not scope_rationale:
        raise LedgerError("in_scope claims require non-empty scope_rationale")
    if not scope_evidence:
        raise LedgerError("in_scope claims require at least one scope_evidence item")


def _normalize_scope_evidence(items: list[dict]) -> list[dict]:
    if not isinstance(items, list):
        raise LedgerError("scope_evidence must be a list")
    normalized = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise LedgerError(f"scope_evidence[{index}] must be an object")
        source = str(item.get("source") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if not source:
            raise LedgerError(f"scope_evidence[{index}].source must be non-empty")
        if not summary:
            raise LedgerError(f"scope_evidence[{index}].summary must be non-empty")
        normalized.append({"source": source, "summary": summary})
    return normalized


def _normalize_claim_scope(scope: ClaimScope | str) -> ClaimScope:
    try:
        return scope if isinstance(scope, ClaimScope) else ClaimScope(scope)
    except ValueError as exc:
        valid = ", ".join(item.value for item in ClaimScope)
        raise LedgerError(f"invalid claim scope {scope!r}; expected one of: {valid}") from exc
