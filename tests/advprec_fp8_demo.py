"""Yardstick for the FP8 softmax red-team entry (precision_verification.md class 3).

Validates the ABSTAIN path. The same tail-wrong kernel is compared two ways:
  - in fp32 the error is real and nonzero (a tight tolerance / downstream metric
    has signal),
  - in FP8 the error is EXACTLY 0 (the tail flushes below FP8 resolution), so no
    tolerance and no input distribution can recover it.

Conclusion: on FP8 outputs the numerical judge would FALSE-ACCEPT; the only
honest verdict is ABSTAIN -> route to a downstream / task-level check. This is
what verifier/classify.py encodes (FP8 output -> low_bit -> J3/abstain).

Run: CUDA_VISIBLE_DEVICES="" uv run python tests/advprec_fp8_demo.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_KPATH = Path(__file__).resolve().parent.parent / "dataset" / "_advprec_softmax_fp8_tail" / "kernel.py"
_spec = importlib.util.spec_from_file_location("_advprec_fp8_kernel", _KPATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
kernel_function = _mod.kernel_function

FP8 = torch.float8_e4m3fn
N = 256
TIGHT_TOL = 1e-4  # a strict tolerance / proxy for a downstream-sensitive judge


def ref_fp32(x):
    return torch.softmax(x, dim=-1)


def peaked(seed: int) -> torch.Tensor:
    """A peaked row whose tail probabilities land ~1e-3 and below — exactly the
    range FP8 flushes to 0."""
    x = torch.full((1, N), -30.0)
    x[0, 0] = 7.0          # peak
    x[0, 1:200] = 0.0      # tail: prob ~ exp(-7)/Z ~ 8e-4, below FP8 resolution
    return x


def err(out, ref):
    return float((out.float() - ref.float()).abs().max())


def main() -> int:
    x = peaked(0)
    r32 = ref_fp32(x)
    k32 = kernel_function(x)

    e_fp32 = err(k32, r32)
    e_fp8 = err(k32.to(FP8), r32.to(FP8))

    print("Red-team FP8 softmax: tail-wrong kernel, judged in fp32 vs FP8\n")
    print(f"  output dtype   max_abs_err   tight-judge (tol={TIGHT_TOL})")
    print(f"  fp32           {e_fp32:9.2e}   {'FAIL' if e_fp32 >= TIGHT_TOL else 'PASS'}"
          "   <- error is real; a tight/downstream judge has signal")
    print(f"  fp8            {e_fp8:9.2e}   {'FAIL' if e_fp8 >= TIGHT_TOL else 'PASS'}"
          "   <- annihilated to 0; NO tolerance/distribution recovers it")
    print(
        "\nReading:\n"
        "  - The bug exists (fp32 error is nonzero), but in FP8 the tail is\n"
        "    structurally unrepresentable, so the error is exactly 0.\n"
        "  - The class-1 rescue (adversarial heavy tail) cannot help: making the\n"
        "    tail matter pushes it into the representable range, a different mode.\n"
        "  - Correct verdict on FP8 output = ABSTAIN (go downstream), not PASS.\n"
        "    verifier/classify.py routes FP8 output -> low_bit -> J3/abstain."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
