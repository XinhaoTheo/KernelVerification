"""Claims-based review loop over a kernel artifact.

Each round:
  author   describes the kernel (context for the skeptic)
  skeptic   files NEW claims (status="open") — concrete, testable assertions
  verifier  runs a probe per open claim, sets status=confirmed/rebutted/inconclusive
Then the judge renders a verdict over the accumulated claims LEDGER.

Convergence: stop when the skeptic files no new claim in a round (nothing left
to test). The ledger is the durable artifact — `claims` in the returned result.
"""

from __future__ import annotations

import os
from typing import Any

from agents import author, judge, skeptic, verifier
from agents.types import Claim, Turn


def run_debate(
    artifact: dict[str, Any],
    *,
    max_rounds: int | None = None,
    tools=None,
    verbose: bool = False,
) -> tuple[dict, list[Turn], list[Claim]]:
    """Run author -> skeptic -> verifier rounds, then judge over the ledger.

    Returns (verdict, full_history, claims_ledger). With verbose=True, prints
    each turn's full text and the verifier's per-claim probe code live.
    """
    if max_rounds is None:
        max_rounds = int(os.getenv("DEBATE_MAX_ROUNDS", "4"))

    history: list[Turn] = []
    ledger: list[Claim] = []

    for round_idx in range(max_rounds):
        auth = author.respond(history, artifact, tools)
        history.append(auth)
        _trace(verbose, auth)

        skp = skeptic.respond(history, artifact, tools)
        history.append(skp)
        new_claims = _register_claims(ledger, skp.get("claims") or [], round_idx)
        _trace(verbose, skp, claims=new_claims)

        # verifier resolves the open claims in place (status + evidence).
        ver = verifier.respond(history, artifact, ledger, tools)
        history.append(ver)
        _trace(verbose, ver, resolved=new_claims)

        if not new_claims:  # skeptic raised nothing new -> converged
            if verbose:
                print(f"\n[debate] round {round_idx}: skeptic filed no new claim → converged")
            break

    verdict = judge.final_verdict(history, artifact, ledger)
    return verdict, history, ledger


def _trace(verbose: bool, turn: Turn, *, claims=None, resolved=None) -> None:
    """Pretty-print one turn (and its structured side-effects) when verbose."""
    if not verbose:
        return
    bar = "─" * 72
    print(f"\n{bar}\n▶ {turn['by'].upper()} (round {turn.get('round')})\n{bar}")
    print(turn.get("text", "").strip())

    if claims is not None:
        print(f"\n  ⤷ filed {len(claims)} new claim(s):")
        for c in claims:
            print(f"     [{c['id']}] {c['type']}: {c['statement'][:100]}")

    if resolved is not None:
        # show the probe code the verifier ran for each claim resolved this round
        for c in resolved:
            ev = [e for e in c.get("evidence", []) if e.get("by") == "verifier"]
            if not ev:
                continue
            last = ev[-1]
            print(f"\n  ⤷ probe for [{c['id']}] → {c.get('status', '?').upper()}")
            print(_indent(last.get("probe_code", "(no code)"), "       "))
            print(f"       probe stdout: {last.get('probe_stdout', '')[:300]}")


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _register_claims(
    ledger: list[Claim], raw_claims: list[dict], round_idx: int
) -> list[Claim]:
    """Assign ids + metadata to freshly-filed claims and append to the ledger.

    Returns the list of newly added claims (empty if the skeptic filed none).
    """
    added: list[Claim] = []
    for raw in raw_claims:
        statement = (raw or {}).get("statement", "").strip()
        if not statement:
            continue
        claim: Claim = {
            "id": f"c{len(ledger) + 1}",
            "type": raw.get("type", "unspecified"),
            "statement": statement,
            "raised_by": "skeptic",
            "round": round_idx,
            "status": "open",
            "evidence": [],
        }
        ledger.append(claim)
        added.append(claim)
    return added
