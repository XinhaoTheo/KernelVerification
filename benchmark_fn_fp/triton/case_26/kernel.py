"""Triton kernel under test: fn16_masked_softmax_fully_masked_row."""
import torch
import triton
import triton.language as tl


@triton.jit
def _masked_softmax_kernel(X, M, Y, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(X + row * stride + cols, mask=valid, other=-float("inf"))
    keep = tl.load(M + row * stride + cols, mask=valid, other=0).to(tl.int1)
    x = tl.where(keep, x, -float("inf"))
    x = x - tl.max(x, axis=0)
    e = tl.where(keep, tl.exp(x), 0.0)
    tl.store(Y + row * stride + cols, e / tl.sum(e, axis=0), mask=valid)


def masked_softmax(x, mask):
    # Row-wise softmax over the positions the mask keeps.
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    _masked_softmax_kernel[(n_rows,)](x, mask, y, x.stride(0), n_cols,
                                      BLOCK=triton.next_power_of_2(n_cols))
    return y
