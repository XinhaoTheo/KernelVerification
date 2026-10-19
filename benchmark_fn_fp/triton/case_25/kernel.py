"""Triton kernel under test: fp10_topk_boundary_tie_admits_all."""
import torch
import triton
import triton.language as tl


@triton.jit
def _threshold_kernel(Scores, Out, stride, K, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(Scores + row * stride + cols, mask=valid, other=-float("inf"))

    lo = tl.min(tl.where(valid, x, float("inf")), axis=0)
    hi = tl.max(x, axis=0)
    for _ in range(40):
        mid = (lo + hi) * 0.5
        count = tl.sum(tl.where(valid & (x >= mid), 1, 0), axis=0)
        lo = tl.where(count >= K, mid, lo)
        hi = tl.where(count >= K, hi, mid)
    # Ties at the cutoff are all admitted.
    keep = valid & (x >= lo)
    tl.store(Out + row * stride + cols, tl.where(keep, x, 0.0), mask=valid)


def topk_mask(scores, k: int):
    # Zero out everything below the top-k cutoff, keeping ties at the cutoff.
    n_rows, n_cols = scores.shape
    out = torch.empty_like(scores)
    _threshold_kernel[(n_rows,)](scores, out, scores.stride(0), k, N=n_cols,
                                 BLOCK=triton.next_power_of_2(n_cols))
    return out
