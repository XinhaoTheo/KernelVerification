"""Judge: neutral arbiter. Decides convergence and renders a final verdict.

MVP convergence: "no new arguments this round" — judge re-reads the latest
author + skeptic turns and decides if either side introduced something
substantively new. Phase 2: switch to "all claims confirmed/rebutted".
"""

from __future__ import annotations

import json
import re

from verifier import llm_client

from .types import Claim, Turn

SYSTEM_NO_NEW = """You are the JUDGE in a structured review of a Triton GPU kernel.

You will be shown the full review so far (author + skeptic turns). Decide
whether the MOST RECENT author and skeptic turns each introduced NEW
substantive content relative to ALL prior turns.

EXPLICIT SENTINELS (strongest signal — check these first):
  - If the latest author turn STARTS WITH `NO_NEW_OBSERVATIONS.`  -> new_author = false
  - If the latest skeptic turn STARTS WITH `NO_NEW_CONCERNS.`     -> new_skeptic = false
The sentinel is authoritative; trust it over heuristics below.

If no sentinel is present, use these criteria. "New" means AT LEAST ONE of:
  - A concern / bug / edge case not previously raised
  - Concrete evidence cited for the FIRST time (not a re-cite of an old line)
  - A new unanswered question
  - A scope-changing refinement (broader/narrower than before)

"New" does NOT mean:
  - Rephrasing or restating prior arguments with fresh wording
  - Adding decorative line citations to already-discussed concerns
  - Reformatting prior conclusions as tables / bullet lists / summaries
  - Acknowledgement ("confirmed", "I agree", "as the author noted")
  - Closing remarks ("I have nothing to add", "let me crystallize")

If a turn is essentially "I confirm/restate what was said", mark it NOT new.

Reply with exactly one JSON object on a single line:
  {"new_author": true|false, "new_skeptic": true|false, "reason": "<one short sentence>"}

Do not include any other text.
"""

SYSTEM_VERDICT = """You are the JUDGE in a structured review of a Triton GPU kernel.

You are given a CLAIMS LEDGER. Each claim was filed by the skeptic and resolved
by the verifier, who EMPIRICALLY ran the kernel on a targeted input:
  - confirmed    = verifier ran it, the kernel is wrong on that input (with numbers)
  - rebutted     = verifier ran it, the kernel handled it correctly
  - inconclusive = verifier could not test it from code
  - open         = never tested

Your job: render a final verdict. You make the SEVERITY call that counting
cannot — in particular you MAY OVERRIDE a "confirmed" if the measured
divergence is actually expected behavior (e.g. ordinary bf16 rounding at a large
reduction dim, not a real bug). When you override, say so in that claim's note.

Decision guidance:
- Any confirmed claim that is a GENUINE defect (not expected precision) → "reject".
- All claims rebutted or overridden-as-benign, nothing real stands → "trust".
- Real concerns remain only inconclusive/open (untested) → "needs_more_evidence".

Reply with exactly one JSON object:
  {
    "verdict": "trust" | "reject" | "needs_more_evidence",
    "confidence": 0.0-1.0,
    "decisive_claims": ["c2", ...],          // claim ids that drove the verdict
    "claim_notes": {"c2": "confirmed and genuine: 256 >> bf16 ulp, real bug"},
    "reason": "<2-4 sentences citing claim ids and their empirical evidence>"
  }

Weigh measured evidence over rhetoric.
"""


_AUTHOR_DONE = "NO_NEW_OBSERVATIONS."
_SKEPTIC_DONE = "NO_NEW_CONCERNS."


def no_new_arguments(history: list[Turn], artifact: dict, round_idx: int) -> bool:
    """Return True if neither side introduced anything new this round.

    Fast path: if both the latest author and skeptic turns START with their
    explicit convergence sentinels, return True without calling the LLM. This
    is the common, cheap case once the conversation has converged.

    Slow path: fall back to a judge LLM call with SYSTEM_NO_NEW prompt that
    checks for the same sentinels plus heuristic criteria.
    """
    if round_idx == 0:
        return False

    latest_author = _latest_by(history, "author")
    latest_skeptic = _latest_by(history, "skeptic")
    if (
        latest_author
        and latest_skeptic
        and latest_author.lstrip().startswith(_AUTHOR_DONE)
        and latest_skeptic.lstrip().startswith(_SKEPTIC_DONE)
    ):
        return True

    raw = llm_client.call(
        system=SYSTEM_NO_NEW,
        artifact=artifact,
        history=history,
        max_tokens=256,
    )
    parsed = _extract_json(raw)
    if not parsed:
        return False
    return not parsed.get("new_author", True) and not parsed.get("new_skeptic", True)


def _latest_by(history: list[Turn], role: str) -> str | None:
    for turn in reversed(history):
        if turn.get("by") == role:
            return turn.get("text", "")
    return None


def final_verdict(history: list[Turn], artifact: dict, ledger: list[Claim] | None = None) -> dict:
    """Render the final verdict over the claims ledger.

    The ledger (skeptic-filed, verifier-resolved claims) is the primary input;
    the prose history is supporting context. Judge can override a "confirmed"
    it deems expected behavior via claim_notes.
    """
    ledger = ledger or []
    system = SYSTEM_VERDICT + "\n\n=== CLAIMS LEDGER ===\n" + _format_ledger(ledger)
    raw = llm_client.call(
        system=system,
        artifact=artifact,
        history=history,
        max_tokens=1024,
    )
    parsed = _extract_json(raw)
    if parsed is None:
        return {
            "verdict": "needs_more_evidence",
            "confidence": 0.0,
            "reason": "judge response could not be parsed",
            "raw": raw,
        }
    return parsed


def _format_ledger(ledger: list[Claim]) -> str:
    if not ledger:
        return "(no claims were filed)"
    lines = []
    for c in ledger:
        ev = "; ".join(
            e.get("verdict_line", "") for e in c.get("evidence", []) if e.get("verdict_line")
        )
        lines.append(
            f"- [{c.get('id', '?')}] status={c.get('status', 'open')} "
            f"type={c.get('type', '?')}\n"
            f"    statement: {c.get('statement', '')}\n"
            f"    evidence:  {ev or '(none)'}"
        )
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
