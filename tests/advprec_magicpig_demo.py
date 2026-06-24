"""Yardstick for the MagicPIG-style sparse-attention red-team entry
(precision_verification.md §4; MagicPIG = arXiv 2410.16179).

The verifier-relevant lesson: judge sparse attention by the DOWNSTREAM output o, not
the selected key set. We compare the LSH-sampling kernel's o against dense attention:

  - benign (random keys): LSH keeps the keys similar to q (the high-score ones), so o
    is close to dense -> looks fine.
  - adversarial (a dominant key whose hash collides into a different bucket): the
    kernel DROPS it, so o loses that key's value mass -> large error. Only visible in o.

Run: CUDA_VISIBLE_DEVICES="" uv run python tests/advprec_magicpig_demo.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_ENTRY = Path(__file__).resolve().parent.parent / "dataset" / "_advprec_magicpig_attn"
_spec = importlib.util.spec_from_file_location("_magicpig_kernel", _ENTRY / "kernel.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
kernel_function = _mod.kernel_function

D = 64
N = 256


def dense(q, k, v):
    w = torch.softmax((k @ q) * (D ** -0.5), dim=-1)
    return w @ v


def err(q, k, v):
    return float((kernel_function(q, k, v) - dense(q, k, v)).abs().max())


def concentrated(seed):
    """LSH's favorable case: a few keys strongly aligned with q (high scores). They are
    SIMILAR to q, so LSH hashes them into q's bucket and keeps them -> o ~ dense."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(D, generator=g)
    k = torch.randn(N, D, generator=g) * 0.2                 # background: near-orthogonal
    v = torch.randn(N, D, generator=g)
    for i in range(4):                                        # a few dominant, q-aligned keys
        k[i] = q / q.norm() * 16.0 + torch.randn(D, generator=g) * 0.1
    return q, k, v


def uniform(seed):
    """The paper's stated failure: 'uniformly distributed attention'. High-dim random
    keys are ~orthogonal to q, so scores are flat and dense attention averages ALL keys;
    LSH keeps only a bucket subset -> the partition function is under-estimated and o is
    off. Distribution-dependent failure."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(D, generator=g),
            torch.randn(N, D, generator=g),
            torch.randn(N, D, generator=g))


def evaluate(name, make, seeds=range(8)):
    e = max(err(*make(s)) for s in seeds)
    tol = 1e-2
    print(f"  {name:<36s} max|o_kernel - o_dense| = {e:7.4f}   "
          f"downstream-judge = {'PASS' if e < tol else 'FAIL'}")
    return e


def main() -> int:
    print("MagicPIG-style LSH sparse attention vs dense attention (judge: downstream o)\n")
    evaluate("concentrated scores (LSH works)", concentrated)
    evaluate("uniform scores (paper's failure mode)", uniform)
    print(
        "\nReading:\n"
        "  - concentrated: the high-score keys are similar to q, so LSH keeps them -> o ~ dense.\n"
        "  - uniform: flat scores -> dense averages ALL keys, but LSH keeps only a bucket\n"
        "    subset, under-estimating the partition function -> o is off. The error is real\n"
        "    and DISTRIBUTION-DEPENDENT (the paper's stated failure), visible only in o.\n"
        "  - Finding: classify labels this attention 'preserve' (the gain probe on a flat\n"
        "    softmax sees a magnitude-preserving average), so precision_recheck SKIPS it and\n"
        "    would miss this bug. Catching MagicPIG needs (a) composite-op classification and\n"
        "    (b) multi-input adversarial support — a real gap this red-team entry exposes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
