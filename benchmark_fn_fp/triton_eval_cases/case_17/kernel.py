import torch
import triton
import triton.language as tl


@triton.jit
def _chunked_scan_kernel(X, OUT, stride_xb, stride_ob, seqlen, CHUNK: tl.constexpr):
    b = tl.program_id(0)
    x_row = X + b * stride_xb
    o_row = OUT + b * stride_ob
    n_chunks = seqlen // CHUNK

    carry = 0.0
    for c in range(n_chunks):
        offs = c * CHUNK + tl.arange(0, CHUNK)
        x = tl.load(x_row + offs, mask=offs < seqlen, other=0.0)
        local = tl.cumsum(x, axis=0)
        tl.store(o_row + offs, local + carry, mask=offs < seqlen)
        carry += tl.sum(x, axis=0)


def chunked_cumsum(x: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Cumulative sum along the sequence axis, carried across fixed-size chunks."""
    batch, seqlen = x.shape
    out = torch.zeros_like(x)
    _chunked_scan_kernel[(batch,)](x, out, x.stride(0), out.stride(0), seqlen, CHUNK=chunk)
    return out
