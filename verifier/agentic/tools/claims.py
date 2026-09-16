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
    # Every field carries its precondition. Without these the preconditions are
    # only discoverable by having a call rejected, and a model that recovers by
    # re-recording the claim instead of fixing the arguments leaves a duplicate
    # behind: one measured run recorded the same claim twice, word for word, and
    # walked the whole probe/finalize/resolve sequence over both copies.
    return {
        "type": "object",
        "required": ["statement", "rationale"],
        "properties": {
            "statement": {
                "type": "string",
                "description": "The falsifiable proposition itself, in one sentence. "
                               "Check read_claim_ledger first: if this restates a claim "
                               "already in the ledger, do not record it again -- add to "
                               "the existing claim with append_evidence instead.",
            },
            "rationale": {
                "type": "string",
                "description": "Why this might be true. Required, and separate from "
                               "scope_rationale.",
            },
            "scope": {
                "type": "string", "enum": _CLAIM_SCOPE_VALUES, "default": "unknown",
                "description": "Use in_scope ONLY when passing scope_rationale AND at "
                               "least one scope_evidence item in this same call; the "
                               "call is rejected otherwise. Leave it unset if you "
                               "cannot yet cite the contract.",
            },
            "scope_rationale": {
                "type": "string", "default": "",
                "description": "Required when scope is in_scope: which contract "
                               "requirement this claim would violate.",
            },
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
                "default": [],
                "description": "Required when scope is in_scope: at least one quote "
                               "from the contract, each with its source.",
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
            "claim_id": {"type": "string", "description": "e.g. c1."},
            "status": {
                "type": "string", "enum": _CLAIM_STATUS_VALUES,
                "description": "The claim must already hold evidence whose `supports` "
                               "equals this status; append_evidence (or "
                               "finalize_probe_evidence) first. If this call is "
                               "rejected for missing evidence, supply the evidence -- "
                               "do NOT retry with a different status, which changes "
                               "your conclusion rather than supporting it.",
            },
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


_DUPLICATE_CLAIM_RATIO = 0.85
# Below this the check is off. Real claims are full propositions -- the measured
# ones run 100-250 characters -- while short strings collide on shape alone:
# "claim 0" and "claim 1" are a 90% textual match and share no meaning at all.
_DUPLICATE_CLAIM_MIN_CHARS = 80


def _reject_duplicate_claim(context: ToolContext, statement: str) -> None:
    """Refuse a claim that restates one already in the ledger.

    A rejected record_claim -- most often for an in_scope claim missing its
    scope_rationale -- is easy to "recover" from by recording the claim again
    with the missing field, which leaves the first copy behind. Measured runs
    did exactly this: 4 duplicate pairs in 18 claims, 2 of them word for word,
    each dragging the whole probe/finalize/resolve sequence through a second
    time for no new information. Every claim also has to be resolved before a
    verdict is allowed, so a duplicate is not merely untidy -- it spends turns
    from the same budget the run needs for real work.

    Matched on normalised text rather than exact equality, because the usual
    second copy is the first one with a clause appended.
    """
    import difflib
    import re

    def norm(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split().__str__()

    if len(statement.strip()) < _DUPLICATE_CLAIM_MIN_CHARS:
        return
    incoming = norm(statement)
    for existing in context.state.claims:
        if len(existing.statement.strip()) < _DUPLICATE_CLAIM_MIN_CHARS:
            continue
        ratio = difflib.SequenceMatcher(None, incoming, norm(existing.statement)).ratio()
        if ratio >= _DUPLICATE_CLAIM_RATIO:
            raise LedgerError(
                f"this restates claim {existing.id}, already in the ledger "
                f"({ratio:.0%} match): {existing.statement[:160]!r}. Do not record it "
                f"again. To add to it use append_evidence(claim_id={existing.id!r}, ...); "
                f"to correct its scope it must be recorded right the first time. If you "
                f"meant a genuinely different proposition, restate it so the difference "
                f"is visible."
            )


def record_claim(context: ToolContext, args: dict) -> dict:
    _enforce_skeptic_claim_limit(context)
    _reject_duplicate_claim(context, str(args["statement"]))
    raised_by = context.current_role or Role.SKEPTIC.value
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
