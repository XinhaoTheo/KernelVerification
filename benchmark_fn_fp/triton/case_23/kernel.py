"""Triton kernel under test: fp9_cosine_similarity_near_zero."""
import torch
import triton
import triton.language as tl


@triton.jit
def _cosine_kernel(A, B, Out, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    a_row = A + row * stride
    b_row = B + row * stride
    dot = 0.0
    na = 0.0
    nb = 0.0
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(a_row + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_row + cols, mask=mask, other=0.0).to(tl.float32)
        dot += tl.sum(a * b, axis=0)
        na += tl.sum(a * a, axis=0)
        nb += tl.sum(b * b, axis=0)
    denom = tl.sqrt(na) * tl.sqrt(nb)
    tl.store(Out + row, dot / tl.maximum(denom, eps))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    """Row-wise cosine similarity of two 2-D float32 tensors."""
    n_rows, n_cols = a.shape
    out = torch.empty(n_rows, device=a.device, dtype=torch.float32)
    _cosine_kernel[(n_rows,)](a, b, out, a.stride(0), n_cols, eps, BLOCK=256)
    return out
