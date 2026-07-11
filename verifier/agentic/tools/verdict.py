"""Verdict state mutation tool for the Judge agent."""

from __future__ import annotations

from pathlib import Path

from verifier.agentic.state import ClaimScope, ClaimStatus, utc_now_iso

from .registry import ToolContext

_VERDICTS = ["trust", "reject", "needs_more_evidence"]


def request_more_debate_schema() -> dict:
    return {
        "type": "object",
        "required": ["reason"],
        "properties": {
            "reason": {"type": "string"},
            "focus_claims": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def request_more_debate(context: ToolContext, args: dict) -> dict:
    reason = str(args["reason"]).strip()
    if not reason:
        raise ValueError("reason must be non-empty")
    focus_claims = args.get("focus_claims", [])
    if not isinstance(focus_claims, list) or not all(isinstance(item, str) for item in focus_claims):
        raise ValueError("focus_claims must be a list of claim ids")

    context.state.convergence = {
        "request": "more_debate",
        "reason": reason,
        "focus_claims": focus_claims,
        "created_at": utc_now_iso(),
    }
    return context.state.convergence


def record_verdict_schema() -> dict:
    return {
        "type": "object",
        "required": ["verdict", "confidence", "decisive_claims", "reason"],
        "properties": {
            "verdict": {"type": "string", "enum": _VERDICTS},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "decisive_claims": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "additionalProperties": False,
    }


def record_verdict(context: ToolContext, args: dict) -> dict:
    verdict = str(args["verdict"])
    if verdict not in _VERDICTS:
        raise ValueError(f"invalid verdict: {verdict}")
    confidence = float(args["confidence"])
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    decisive_claims = args["decisive_claims"]
    if not isinstance(decisive_claims, list) or not all(isinstance(item, str) for item in decisive_claims):
        raise ValueError("decisive_claims must be a list of claim ids")
    reason = str(args["reason"]).strip()
    if not reason:
        raise ValueError("reason must be non-empty")
    _enforce_reject_scope_guard(context, verdict=verdict, decisive_claims=decisive_claims)

    context.state.verdict = {
        "verdict": verdict,
        "confidence": confidence,
        "decisive_claims": decisive_claims,
        "reason": reason,
        "created_at": utc_now_iso(),
    }
    context.state.convergence = None
    return context.state.verdict


def _enforce_reject_scope_guard(context: ToolContext, *, verdict: str, decisive_claims: list[str]) -> None:
    if verdict != "reject":
        return
    if not decisive_claims:
        raise ValueError("reject verdict requires at least one decisive in-scope confirmed claim")
    claims_by_id = {claim.id: claim for claim in context.state.claims}
    invalid = []
    for claim_id in decisive_claims:
        claim = claims_by_id.get(claim_id)
        if claim is None:
            invalid.append(f"{claim_id}: unknown claim")
            continue
        status = claim.status.value if hasattr(claim.status, "value") else str(claim.status)
        scope = claim.scope.value if hasattr(claim.scope, "value") else str(claim.scope)
        if status != ClaimStatus.CONFIRMED.value or scope != ClaimScope.IN_SCOPE.value:
            invalid.append(f"{claim_id}: status={status}, scope={scope}")
            continue
        if not claim.scope_evidence:
            invalid.append(f"{claim_id}: missing scope_evidence")
            continue
        if not _has_benchmark_scope_evidence(context, claim.scope_evidence):
            invalid.append(f"{claim_id}: missing benchmark/test-domain scope_evidence")
    if invalid:
        raise ValueError(
            "reject verdict can only use decisive claims that are confirmed, in_scope, "
            "and have benchmark/test-domain scope_evidence; "
            + "; ".join(invalid)
        )


def _has_benchmark_scope_evidence(context: ToolContext, scope_evidence: list[dict]) -> bool:
    sources = [str(item.get("source") or "").lower() for item in scope_evidence if isinstance(item, dict)]
    benchmark_markers = (
        "test.py",
        "get_inputs",
        "get_init_inputs",
        "meta.json",
        "metadata",
        "artifact metadata",
        "seed_",
        "seed-",
        "seed file",
    )
    if any(any(marker in source for marker in benchmark_markers) for source in sources):
        return True

    if context.dataset_dir is not None and context.state.entry:
        entry_dir = Path(context.dataset_dir) / str(context.state.entry)
        if entry_dir.exists() and not (entry_dir / "test.py").exists():
            return any("problem.txt" in source for source in sources)
    return False
