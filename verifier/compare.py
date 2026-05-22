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
