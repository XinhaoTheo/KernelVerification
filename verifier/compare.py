"""Fixed correctness comparison — the single source of truth for "does the
kernel output match the reference".

Used by BOTH the recheck test and the verifier probe (copied into their
subprocess workdir as `kverify_compare.py`). The point is that the pass/fail
verdict — dtype-aware tolerance, NaN/inf handling — is decided HERE, once,
not re-invented (badly) by the LLM in every generated probe.

Rules:
  - tolerance by dtype: fp32 -> 1e-3, fp16/bf16 -> 1e-2/2e-2
  - equal_nan=True: kernel and reference both NaN at the same place is a MATCH,
    not a bug (e.g. softmax of all-inf legitimately yields NaN in PyTorch too).
  - a bug is only "kernel DIVERGES from reference", never "kernel produced a
    NaN/large value" in isolation.
"""

from __future__ import annotations

import torch


def compare_outputs(out, ref) -> tuple[bool, float, str]:
    """Return (matches, max_abs_diff, detail).

    matches=True means the kernel output agrees with the reference within the
    dtype-appropriate tolerance. Callers print CONFIRMED (bug) when NOT matches,
    REFUTED when matches.
    """
    if out is None:
        return False, float("inf"), "kernel returned None"
    if not torch.is_tensor(out):
        return False, float("inf"), f"kernel returned non-tensor {type(out).__name__}"
    if tuple(out.shape) != tuple(ref.shape):
        return False, float("inf"), f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}"

    of = out.float()
    rf = ref.float()

    dt = out.dtype
    if dt in (torch.float16, torch.bfloat16):
        rtol, atol = 1e-2, 2e-2
    else:
        rtol, atol = 1e-3, 1e-3

    matches = bool(torch.allclose(of, rf, rtol=rtol, atol=atol, equal_nan=True))

    # NaN/inf-pattern mismatch is itself a divergence (one side NaN, other not).
    nan_mismatch = bool((of.isnan() != rf.isnan()).any().item()) if of.numel() else False
    inf_mismatch = bool((of.isinf() != rf.isinf()).any().item()) if of.numel() else False

    finite = of.isfinite() & rf.isfinite()
    if finite.any():
        max_diff = float((of[finite] - rf[finite]).abs().max().item())
    else:
        max_diff = 0.0

    detail = (
        f"dtype={dt}, rtol={rtol}, atol={atol}, "
        f"max_finite_diff={max_diff:.6g}, "
        f"nan_mismatch={nan_mismatch}, inf_mismatch={inf_mismatch}, match={matches}"
    )
    return matches, max_diff, detail


# --------------------------------------------------------------------------- #
# Class-routed judging (precision_verification.md §5.2)                        #
#                                                                             #
# `compare_outputs` above IS the numerical judge (J1) and stays the default   #
# for continuous ops. The functions below add the other judges and a router   #
# so the right one is chosen by operator class (from verifier/classify.py),   #
# instead of forcing one allclose tolerance onto every operator.              #
# --------------------------------------------------------------------------- #

# J2 default: a dropped key is "harmful" if it scores more than this above the
# smallest key the candidate kept. ~0 means a harmless boundary swap.
DEFAULT_GAP_TOL = 0.05


def judge_selection(cand_idx, ref_idx, scores, *, gap_tol: float = DEFAULT_GAP_TOL):
    """J2 — judge a top-k/selection by VALUE-GAP, never by raw index recall.

    Index recall punishes harmless tie-break swaps (and is non-deterministic
    under races), so it both false-rejects correct kernels and fails to rank
    harm. Instead: for each TRUE top-k key the candidate dropped, measure how
    much better it is than the worst key the candidate kept. gap≈0 => harmless
    swap (kept something as good); gap big => dropped a clearly-better key.
    """
    import torch
    s = torch.as_tensor(scores).flatten().float()
    cset = {int(i) for i in torch.as_tensor(cand_idx).flatten().tolist()}
    rset = {int(i) for i in torch.as_tensor(ref_idx).flatten().tolist()}
    missed = rset - cset
    recall = len(cset & rset) / max(len(rset), 1)
    if not missed:
        return True, {"verdict": "match", "judge": "J2", "value_gap": 0.0,
                      "recall": recall}
    min_kept = min((s[i].item() for i in cset), default=float("-inf"))
    gap = max(s[m].item() - min_kept for m in missed)
    matches = gap <= gap_tol
    return matches, {
        "verdict": "match" if matches else "mismatch", "judge": "J2",
        "value_gap": round(gap, 4), "recall": round(recall, 3),
        "note": "value-gap weighted; index recall is reported but NOT the verdict",
    }


def abstain(reason: str) -> dict:
    return {"verdict": "abstain", "judge": "J3/abstain", "reason": reason}


def judge(out, ref, *, op_class: str = "preserve", scores=None,
          gap_tol: float = DEFAULT_GAP_TOL) -> dict:
    """Route to the judge that fits the operator class (classify.py's label).

    Returns {verdict: "match"|"mismatch"|"abstain", judge, ...}. "abstain" is a
    first-class outcome: when the judge structurally cannot tell a correct kernel
    from a buggy one (low-bit outputs), reporting abstain is correct, a confident
    wrong PASS/FAIL is not.
    """
    if op_class == "low_bit":
        return abstain(
            "output dtype is low-bit (fp8/fp4/int4): sub-resolution errors round "
            "to 0, so numerical comparison cannot separate correct from buggy. "
            "Route to a downstream / task-level check."
        )
    if op_class == "select":
        if scores is None:
            return abstain("selection judge (J2) needs the input scores to weight "
                           "dropped keys by value-gap")
        _, detail = judge_selection(out, ref, scores, gap_tol=gap_tol)
        return detail
    # preserve / compress -> numerical judge (J1). For compress, the operator
    # class still matters upstream: recheck must FEED adversarial distributions
    # (that is pillar X, not the judge), but the judge itself is allclose.
    matches, max_diff, detail = compare_outputs(out, ref)
    return {"verdict": "match" if matches else "mismatch", "judge": "J1",
            "max_diff": max_diff, "detail": detail}
