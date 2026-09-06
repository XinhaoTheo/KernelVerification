import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import count_tied_at_boundary as count_buggy  # real primitive, eps=1e-3

import triton
import triton.language as tl


@triton.jit
def _update_min_larger_stats_ref(data, above_mask, min_larger, num_min_larger, sentinel, EPS: tl.constexpr):
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < EPS)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < EPS
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _count_tied_at_boundary_kernel_ref(scores_ptr, count_ptr, N: tl.constexpr, pivot, EPS: tl.constexpr):
    offs = tl.arange(0, N)
    data = tl.load(scores_ptr + offs)
    above_mask = data > pivot
    min_larger = tl.full((), float("inf"), tl.float32)
    num_min_larger = tl.zeros((), tl.int32)
    min_larger, num_min_larger = _update_min_larger_stats_ref(data, above_mask, min_larger, num_min_larger, float("inf"), EPS)
    tl.store(count_ptr, num_min_larger)


def count_tied_at_boundary_reference(scores, pivot):
    N = scores.shape[0]
    count = torch.empty(1, dtype=torch.int32, device=scores.device)
    _count_tied_at_boundary_kernel_ref[(1,)](scores, count, N=N, pivot=pivot, EPS=1e-9)  # real epsilon
    return int(count.item())


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"

    # One clear top value, then a tight cluster of 3 distinct values near the
    # boundary: the smallest of the cluster (5.0) is separated from the other
    # two by 3e-4 and 6e-4 -- well outside the real kernel's 1e-9 tolerance,
    # but well inside a (still reasonable) 1e-3 tolerance.
    N = 16
    scores = torch.full((N,), -10.0, device=device)
    scores[0] = 10.0
    scores[1] = 5.0
    scores[2] = 5.0003
    scores[3] = 5.0006
    pivot = 0.0

    ref_count = count_tied_at_boundary_reference(scores, pivot)  # real eps=1e-9: only the exact minimum
    cand_count = count_buggy(scores, pivot)  # eps=1e-3: the whole near-boundary cluster

    assert ref_count != cand_count, "the two epsilons should disagree on the boundary tie count"

    # Downstream use (as a real multi-pass top-k kernel would): split a fixed
    # unit of remaining "selection budget" equally among the tied boundary
    # candidates.
    ref_share = 1.0 / ref_count
    cand_share = 1.0 / cand_count
    naive_pass = abs(ref_share - cand_share) < 1e-2

    print(f"real vLLM tie-counting primitive (GPU): reference (eps=1e-9) count={ref_count}, candidate (eps=1e-3) count={cand_count}")
    print(f"downstream per-candidate share: reference={ref_share:.4f}, candidate={cand_share:.4f}, naive allclose-style check={naive_pass} (expected False)")
    print("FP DEMONSTRATED" if not naive_pass else "tune constants")
    # What a conventional CI test would conclude: 拿一个实现当唯一参考做 allclose
    print(f"NAIVE_ALLCLOSE_VERDICT: {naive_pass}")
    return not naive_pass


if __name__ == "__main__":
    test_kernel()
