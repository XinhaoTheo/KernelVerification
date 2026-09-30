"""Triton kernel under test: fp11_fp8_e4m3_rounding_within_format."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_roundtrip_kernel(X, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)

    # e4m3: 3 mantissa bits, so quantize to 1/8 of a binade.
    ax = tl.abs(x)
    e = tl.floor(tl.log2(tl.where(ax == 0.0, 1.0, ax)))
    step = tl.exp2(e - 3.0)
    q = tl.floor(x / step + 0.5) * step
    q = tl.where(ax == 0.0, 0.0, q)
    tl.store(OUT + offs, q, mask=mask)


def fp8_roundtrip(x):
    # Round a float32 tensor to the nearest value representable in e4m3.
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    _fp8_roundtrip_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out
