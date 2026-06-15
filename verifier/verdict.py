"""Finalize the verdict — a THIN deterministic floor over the judge (design A).

The multi-agent JUDGE is the final arbiter. It already weighs ALL the signals
itself (precision_verification.md): standard recheck status + robustness are put
in its prompt ([agents/judge.py](agents/judge.py) `_decision_context`), and
precision_recheck's empirical finding enters its claims ledger as a pre-filed
claim (design B). So the judge's verdict — trust / reject / needs_more_evidence /
needs_downstream — is the decision.

`combine` does NOT re-decide on top of the judge (that would make the judge's
reasoning over the same signals pointless). It only enforces ONE fact-based floor
that is not a judgment call: a kernel that is wrong on the problem's NORMAL inputs
cannot be reported as "trust", whatever the judge said. Robustness gaps are folded
in as a confidence note, never a veto.
"""

from __future__ import annotations

_VERDICTS = ("trust", "reject", "needs_more_evidence", "needs_downstream")


def combine(*, recheck_status: str, debate_verdict: str | None,
            robustness: dict | None = None) -> dict:
    """Return {final, reason, [robustness_gaps, note]}. Defers to the judge;
    applies only the standard-correctness floor."""
    robustness = robustness or {}
    rob_fails = [k for k, v in robustness.items() if v == "fail"]

    final = debate_verdict if debate_verdict in _VERDICTS else "needs_more_evidence"
    reason = f"judge (final arbiter) verdict = {debate_verdict}"

    # The one floor: standard correctness is a fact, not a severity call.
    if recheck_status == "failed" and final == "trust":
        final = "reject"
        reason = "safety floor: standard recheck failed → cannot be trusted (judge said trust)"

    out = {"final": final, "reason": reason}
    if rob_fails:
        out["robustness_gaps"] = rob_fails
        out["note"] = "robustness gaps present (not a veto); lowers confidence"
    return out
