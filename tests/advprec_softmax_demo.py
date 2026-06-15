"""Yardstick for the softmax red-team entry (precision_verification.md class 1).

Demonstrates the compression-driven false accept with NUMBERS:
  - benign torch.randn -> the tail-truncating kernel matches the reference within
    tolerance (the omitted tail mass is ~0), so a benign-only recheck ACCEPTS it.
  - adversarial (a peak + a heavy band of comparable keys just past the cutoff)
    -> the dropped tail mass is real, every kept weight is inflated, and the
    max error is far above tolerance.

Run: CUDA_VISIBLE_DEVICES="" uv run python tests/advprec_softmax_demo.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_KPATH = Path(__file__).resolve().parent.parent / "dataset" / "_advprec_softmax_tail" / "kernel.py"
_spec = importlib.util.spec_from_file_location("_advprec_softmax_kernel", _KPATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
kernel_function = _mod.kernel_function

N = 256
TOL = 2e-2  # the tolerance the real recheck uses for these (bf16) kernels


def reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def max_abs_err(x: torch.Tensor) -> float:
    return float((kernel_function(x) - reference(x)).abs().max())


def benign(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, N, generator=g)


def adversarial(seed: int) -> torch.Tensor:
    """A peak plus a heavy band of comparable keys placed just past the cutoff
    (shifted ≈ -8.5). Each weighs ~exp(-8.5), but ~200 of them carry real
    aggregate mass that the kernel silently drops."""
    x = torch.full((1, N), -30.0)   # inert filler
    x[0, 0] = 8.5                    # the peak (kept by the kernel)
    x[0, 1:200] = 0.0                # band: shifted = -8.5, just past cutoff -8
    return x


def evaluate(name: str, make, seeds=range(8)) -> float:
    errs = [max_abs_err(make(s)) for s in seeds]
    e = max(errs)
    print(f"  {name:<26s} max_abs_err={e:7.4f}   numerical-judge={'PASS' if e < TOL else 'FAIL'}")
    return e


def main() -> int:
    print("Red-team softmax: tail-truncating kernel vs exact softmax "
          f"(tol={TOL})\n")
    evaluate("benign (randn)", benign)
    evaluate("adversarial (heavy tail)", adversarial)
    print(
        "\nReading:\n"
        "  - benign: error ~0 -> softmax COMPRESSES the dropped tail to nothing,\n"
        "    so a recheck that only tests randn ACCEPTS this kernel.\n"
        "  - adversarial: the same kernel is off by a wide margin once the tail\n"
        "    carries real mass. Only the adversarial distribution (pillar X)\n"
        "    exposes it; the bug was hidden by compression, not absent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
