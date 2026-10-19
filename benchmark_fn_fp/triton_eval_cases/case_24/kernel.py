import torch
import triton
import triton.language as tl


@triton.jit
def _requant_step_kernel(X, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.where(absmax == 0.0, 1.0, absmax / 127.0)
    q = tl.floor(x / scale).to(tl.int8)
    tl.store(OUT + offs, q.to(tl.float32) * scale, mask=mask)


def requantize(x: torch.Tensor) -> torch.Tensor:
    """One symmetric INT8 quantize-dequantize round trip over a 1-D tensor."""
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    _requant_step_kernel[(triton.cdiv(n, BLOCK),)](x, out, n, BLOCK=BLOCK)
    return out
