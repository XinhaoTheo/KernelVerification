"""Yardstick for the topk red-team entry (precision_verification.md class 2).

Demonstrates the distribution-dependent false accept with NUMBERS:
  - benign torch.randn  -> the flawed kernel's recall ≈ 100%, so a recheck that
    only tests randn ACCEPTS it.
  - adversarial (winners concentrated in one block) -> recall collapses AND the
    missed keys are large-valued (big value-gap = HARMFUL, not a boundary swap).

It also contrasts two judges on the same outputs:
  - recall ≥ 0.95 (the naive surrogate metric)
  - J2 value-gap (penalty = how much larger a dropped key is than the smallest
    key the kernel kept) — the note's "不是所有 miss 都一样".

Run: CUDA_VISIBLE_DEVICES="" uv run python tests/advprec_topk_demo.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_KPATH = Path(__file__).resolve().parent.parent / "dataset" / "_advprec_topk_boundary" / "kernel.py"
_spec = importlib.util.spec_from_file_location("_advprec_topk_kernel", _KPATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
kernel_function = _mod.kernel_function

K = 8
N = 256


def reference(scores: torch.Tensor) -> torch.Tensor:
    return torch.topk(scores, K, dim=-1).indices


def recall(cand: torch.Tensor, ref: torch.Tensor) -> float:
    c, r = set(cand.flatten().tolist()), set(ref.flatten().tolist())
    return len(c & r) / len(r)


def value_gap(scores: torch.Tensor, cand: torch.Tensor, ref: torch.Tensor) -> float:
    """J2: max over dropped true-top-k keys of (its value - smallest kept value).

    gap≈0  => harmless boundary swap (kept something nearly as good).
    gap big => HARMFUL (dropped a clearly-better key for a worse one).
    """
    s = scores.flatten()
    cset, rset = set(cand.flatten().tolist()), set(ref.flatten().tolist())
    missed = rset - cset
    if not missed:
        return 0.0
    min_kept = min(s[i].item() for i in cset)
    return max(s[m].item() - min_kept for m in missed)


def benign(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, N, generator=g)


def adversarial(seed: int) -> torch.Tensor:
    """The true top-8 keys all sit in block 0 (indices 0..7) — winners concentrate
    in one block. A real analogue: attention sinks / recent tokens dominate."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(1, N, generator=g)          # background ~ N(0,1)
    scores[0, 0:K] = torch.linspace(6.0, 5.3, K)     # 8 dominant winners, one block
    return scores


def evaluate(name: str, make, seeds=range(8)):
    recalls, gaps = [], []
    for s in seeds:
        scores = make(s)
        cand = kernel_function(scores, K)
        ref = reference(scores)
        recalls.append(recall(cand, ref))
        gaps.append(value_gap(scores, cand, ref))
    r = sum(recalls) / len(recalls)
    gmax = max(gaps)
    recall_pass = r >= 0.95
    j2_pass = gmax < 0.05
    print(f"  {name:<26s} recall={r:5.2f}  max_value_gap={gmax:6.3f}   "
          f"recall-judge={'PASS' if recall_pass else 'FAIL'}   "
          f"J2-judge={'PASS' if j2_pass else 'FAIL'}")
    return r, gmax


def main() -> int:
    print("Red-team topk: per-block-quota approximate kernel vs exact top-k\n")
    print("  distribution               metric                       verdicts")
    evaluate("benign (randn)", benign)
    evaluate("adversarial (concentrated)", adversarial)
    print(
        "\nReading:\n"
        "  - benign: recall≈1.0 -> a recheck that ONLY tests randn ACCEPTS this kernel.\n"
        "  - adversarial: recall collapses AND value-gap is large -> the kernel\n"
        "    silently drops the most important keys. Only visible once you feed the\n"
        "    adversarial distribution (the third pillar X), not benign randn.\n"
        "  - value-gap (J2) confirms the misses are HARMFUL, not harmless swaps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
