"""Verifier: empirically resolves each OPEN claim by running the kernel.

Roadmap direction D (separate role) + B (structured claims): the skeptic files
claims (status="open"); the verifier walks the open claims, writes a targeted
probe for each, runs it on GPU, and sets the claim's status to confirmed /
rebutted / inconclusive with measured evidence. It reports MEASUREMENTS and a
tentative label — the judge keeps the final say on severity (it can override a
"confirmed" it deems expected, e.g. bf16 rounding).

The verifier mutates the ledger claims in place (status + evidence).
"""

from __future__ import annotations

from pathlib import Path

from verifier import llm_client, recheck

from .types import Claim, Turn

# Cap probes per round so a flood of claims can't blow up cost/time.
MAX_CLAIMS_PER_ROUND = 5

SYSTEM = """You are the VERIFIER in a structured review of a Triton GPU kernel.

You are given the problem spec, the kernel code, and ONE claim to test. The
claim asserts the kernel may be wrong on a specific input. Your job is to test
it EMPIRICALLY, not argue.

Write a SHORT standalone Python probe that:
  - `from kernel import kernel_function`
  - `from kverify_compare import compare_outputs`  (provided helper, always available)
  - Constructs the EXACT input named in the claim (non-contiguous slice,
    non-aligned size, specific dtype, large reduction dim, etc.).
  - Computes a PyTorch reference for that input, in the dtype the kernel expects
    (if the kernel asserts a dtype, build inputs in it).
  - Calls `matches, max_diff, detail = compare_outputs(out, ref)` to decide
    correctness. DO NOT invent your own tolerance or threshold, and DO NOT
    flag a NaN/inf in isolation — `compare_outputs` already handles dtype-aware
    tolerance and treats matching NaN/inf as correct (a bug is only when the
    kernel DIVERGES from the reference).
  - Prints, as the FINAL line, exactly one of:
      "CONFIRMED: <phrase>"    matches is False  -> kernel wrong (claim TRUE)
      "REFUTED: <phrase>"      matches is True   -> kernel correct (claim FALSE)
      "INCONCLUSIVE: <why>"    genuinely cannot be tested from code
    and print `detail` on a preceding line.
  - Wrap the body in try/except so a crash prints "CONFIRMED: kernel raised
    <ExcType>: <msg>" — a crash on a VALID input is itself a real defect.
    (But if the kernel deliberately ASSERTS a precondition the claim's input
    violates — e.g. asserts contiguity — that is a defensive guard, print
    "REFUTED: kernel correctly rejects out-of-contract input via <assert>".)
  - End with `if __name__ == "__main__":` that runs it.

Typical shape:
    out = kernel_function(...)
    ref = <pytorch reference>
    matches, max_diff, detail = compare_outputs(out, ref)
    print(detail)
    print(("REFUTED: " if matches else "CONFIRMED: ") + "<phrase>")

Be faithful: do NOT rig the probe. Output ONLY the probe code in a single
```python fenced block.
"""


def respond(
    history: list[Turn],
    artifact: dict,
    ledger: list[Claim],
    tools=None,
) -> Turn:
    round_idx = sum(1 for t in history if t.get("by") == "verifier")
    open_claims = [c for c in ledger if c.get("status") == "open"]

    if not open_claims:
        return Turn(
            by="verifier",
            round=round_idx,
            text="No open claims to verify this round.",
        )

    kernel_code = artifact.get("kernel_code", "")
    if not kernel_code.strip():
        return Turn(by="verifier", round=round_idx, text="No kernel code to probe.")

    problem = _read_problem(artifact.get("session_dir"))
    summaries: list[str] = []

    for claim in open_claims[:MAX_CLAIMS_PER_ROUND]:
        probe = _gen_probe(problem, kernel_code, claim.get("statement", ""))
        _, stdout, stderr = recheck.run_test(kernel_code, probe)
        status, note = _interpret(stdout, stderr)

        claim["status"] = status
        claim.setdefault("evidence", []).append(
            {
                "by": "verifier",
                "verdict_line": note,
                "probe_code": probe,
                "probe_stdout": stdout.strip()[:1500],
                "probe_stderr": stderr.strip()[:800],
            }
        )
        summaries.append(f"[{claim.get('id', '?')}] {status.upper()}: {note}")

    text = "I ran a targeted probe for each open claim:\n" + "\n".join(summaries)
    return Turn(by="verifier", round=round_idx, text=text)


def _gen_probe(problem: str, kernel_code: str, statement: str) -> str:
    user = (
        "=== PROBLEM SPEC ===\n"
        f"{problem}\n\n"
        "=== KERNEL CODE (exports kernel_function) ===\n"
        f"{kernel_code}\n\n"
        "=== CLAIM TO TEST ===\n"
        f"{statement}\n"
    )
    raw = llm_client.oneshot(SYSTEM, user, max_tokens=2048)
    return recheck._extract_code(raw)


def _interpret(stdout: str, stderr: str) -> tuple[str, str]:
    """Map probe output to (status, one-line note)."""
    for marker, status in (
        ("CONFIRMED", "confirmed"),
        ("REFUTED", "rebutted"),
        ("INCONCLUSIVE", "inconclusive"),
    ):
        line = _last_line_with(stdout, marker)
        if line:
            return status, line
    if stderr.strip():
        # Probe itself errored (not the kernel) — can't conclude.
        return "inconclusive", "probe error: " + stderr.strip()[-200:]
    return "inconclusive", "no verdict line in probe output"


def _last_line_with(text: str, marker: str) -> str | None:
    hit = None
    for line in text.splitlines():
        if marker in line:
            hit = line.strip()
    return hit


def _read_problem(session_dir: str | None) -> str:
    if not session_dir:
        return ""
    p = Path(session_dir) / "problem.txt"
    return p.read_text() if p.exists() else ""
