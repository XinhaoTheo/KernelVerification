"""Triton kernel under test: fn8_fused_softmax_block_tail."""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(OUT, IN, stride_om, stride_im, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    in_row = IN + row * stride_im
    out_row = OUT + row * stride_om
    n_full = n_cols // BLOCK

    row_max = -float("inf")
    for b in range(tl.cdiv(n_cols, BLOCK)):
        cols = b * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    denom = 0.0
    for b in range(n_full):
        cols = b * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=-float("inf"))
        denom += tl.sum(tl.exp(x - row_max), axis=0)

    for b in range(tl.cdiv(n_cols, BLOCK)):
        cols = b * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(in_row + cols, mask=cols < n_cols, other=-float("inf"))
        tl.store(out_row + cols, tl.exp(x - row_max) / denom, mask=cols < n_cols)


def softmax(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Row-wise softmax of a 2-D float32 tensor."""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](out, x, out.stride(0), x.stride(0), n_cols, BLOCK=block_size)
    return out
