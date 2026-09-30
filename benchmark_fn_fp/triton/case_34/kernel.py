"""Triton kernel under test: fn22_fp32_atomic_bitwise."""
import torch
import triton
import triton.language as tl


@triton.jit
def _blocked_sum_kernel(X, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + idx, mask=idx < N, other=0.0)
    tl.atomic_add(Out, tl.sum(x, axis=0))


def blocked_sum(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """Sum of a float32 tensor, one atomic accumulation per block."""
    out = torch.zeros(1, device=x.device, dtype=torch.float32)
    _blocked_sum_kernel[(triton.cdiv(x.numel(), block),)](x, out, x.numel(), BLOCK=block)
    return out
