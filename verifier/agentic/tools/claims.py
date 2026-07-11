"""Claim ledger tools."""

from __future__ import annotations

from verifier.agentic.ledger import ClaimLedger, LedgerError
from verifier.agentic.state import ClaimScope, ClaimStatus, EvidenceKind, Role, utc_now_iso

from .registry import ToolContext

_CLAIM_SCOPE_VALUES = [scope.value for scope in ClaimScope]
_CLAIM_STATUS_VALUES = [status.value for status in ClaimStatus]
_EVIDENCE_SUPPORT_VALUES = [
    ClaimStatus.CONFIRMED.value,
    ClaimStatus.REBUTTED.value,
    ClaimStatus.INCONCLUSIVE.value,
]
_EVIDENCE_KIND_VALUES = [kind.value for kind in EvidenceKind]
_MAX_SKEPTIC_CLAIMS_PER_TURN = 3


def record_claim_schema() -> dict:
    return {
        "type": "object",
        "required": ["statement", "rationale"],
        "properties": {
            "statement": {"type": "string"},
            "rationale": {"type": "string"},
            "scope": {"type": "string", "enum": _CLAIM_SCOPE_VALUES, "default": "unknown"},
            "scope_rationale": {"type": "string", "default": ""},
            "scope_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["source", "summary"],
                    "properties": {
                        "source": {"type": "string"},
                        "summary": {"type": "string"}
                    },
                    "additionalProperties": False
                },
                "default": []
            },
            "raised_by": {
                "type": "string",
                "enum": ["describer", "skeptic", "experimenter", "judge", "orchestrator"],
                "default": "skeptic",
            },
        },
        "additionalProperties": False,
    }


def read_claim_ledger_schema() -> dict:
    return {
        "type": "object",
        "required": [],
        "properties": {},
        "additionalProperties": False,
    }


def append_evidence_schema() -> dict:
    return {
        "type": "object",
        "required": ["claim_id", "kind", "summary", "supports"],
        "properties": {
            "claim_id": {"type": "string"},
            "kind": {"type": "string", "enum": _EVIDENCE_KIND_VALUES},
            "summary": {"type": "string"},
            "supports": {"type": "string", "enum": _EVIDENCE_SUPPORT_VALUES},
            "tool_event_id": {"type": ["string", "null"]},
            "data": {"type": ["object", "null"]},
        },
        "additionalProperties": False,
    }


def update_claim_status_schema() -> dict:
    return {
        "type": "object",
        "required": ["claim_id", "status"],
        "properties": {
            "claim_id": {"type": "string"},
            "status": {"type": "string", "enum": _CLAIM_STATUS_VALUES},
        },
        "additionalProperties": False,
    }


def record_no_new_claims_schema() -> dict:
    return {
        "type": "object",
        "required": ["reason", "reviewed_claims"],
        "properties": {
            "reason": {"type": "string"},
            "reviewed_claims": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def record_claim(context: ToolContext, args: dict) -> dict:
    _enforce_skeptic_claim_limit(context)
    raised_by = str(args.get("raised_by") or Role.SKEPTIC.value)
    claim = ClaimLedger(context.state).record_claim(
        statement=str(args["statement"]),
        rationale=str(args["rationale"]),
        raised_by=raised_by,
        scope=str(args.get("scope") or ClaimScope.UNKNOWN.value),
        scope_rationale=str(args.get("scope_rationale") or ""),
        scope_evidence=args.get("scope_evidence") or [],
    )
    return claim.to_dict()


def read_claim_ledger(context: ToolContext, args: dict) -> dict:
    _ = args
    return {"claims": ClaimLedger(context.state).read_claims()}


def append_evidence(context: ToolContext, args: dict) -> dict:
    data = args.get("data") or {}
    if not isinstance(data, dict):
        raise LedgerError("evidence data must be an object")

    evidence = ClaimLedger(context.state).append_evidence(
        claim_id=str(args["claim_id"]),
        kind=str(args["kind"]),
        summary=str(args["summary"]),
        supports=str(args["supports"]),
        tool_event_id=_optional_str(args.get("tool_event_id")),
        data=data,
    )
    return evidence.to_dict()


def update_claim_status(context: ToolContext, args: dict) -> dict:
    claim = ClaimLedger(context.state).update_claim_status(
        claim_id=str(args["claim_id"]),
        status=str(args["status"]),
    )
    return claim.to_dict()


def record_no_new_claims(context: ToolContext, args: dict) -> dict:
    if context.current_role != Role.SKEPTIC.value:
        raise LedgerError("record_no_new_claims can only be called by the skeptic")
    reviewed_claims = args.get("reviewed_claims") or []
    if not isinstance(reviewed_claims, list) or not all(isinstance(item, str) for item in reviewed_claims):
        raise LedgerError("reviewed_claims must be a list of claim ids")

    review = {
        "decision": "no_new_claims",
        "reason": str(args["reason"]),
        "reviewed_claims": reviewed_claims,
        "reviewed_tool_event_count": len(context.state.tool_events),
        "turn": context.current_turn,
        "tool_event_id": context.current_tool_event_id,
        "created_at": utc_now_iso(),
    }
    context.state.skeptic_review = review
    return review


def _enforce_skeptic_claim_limit(context: ToolContext) -> None:
    if context.current_role != Role.SKEPTIC.value or context.current_turn is None:
        return
    attempted_claims = context.current_turn_tool_counts.get("record_claim", 0)
    if attempted_claims > _MAX_SKEPTIC_CLAIMS_PER_TURN:
        raise LedgerError(
            f"skeptic can record at most {_MAX_SKEPTIC_CLAIMS_PER_TURN} claims per turn"
        )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    return str(value)
