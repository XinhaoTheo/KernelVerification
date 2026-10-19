"""Triton kernel under test: fp8_splitk_atomic_accumulation_order."""
import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_dot_kernel(A, B, Out, K, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    part = tl.program_id(0)
    chunk = tl.cdiv(K, SPLIT)
    start = part * chunk
    acc = 0.0
    for off in range(0, chunk, BLOCK):
        idx = start + off + tl.arange(0, BLOCK)
        mask = idx < tl.minimum(start + chunk, K)
        a = tl.load(A + idx, mask=mask, other=0.0)
        b = tl.load(B + idx, mask=mask, other=0.0)
        acc += tl.sum(a * b, axis=0)
    tl.atomic_add(Out, acc)


def splitk_dot(a: torch.Tensor, b: torch.Tensor, split: int = 512) -> torch.Tensor:
    """Inner product of two float32 vectors, accumulated across `split` partitions."""
    out = torch.zeros(1, device=a.device, dtype=torch.float32)
    _splitk_dot_kernel[(split,)](a, b, out, a.numel(), SPLIT=split, BLOCK=256)
    return out
