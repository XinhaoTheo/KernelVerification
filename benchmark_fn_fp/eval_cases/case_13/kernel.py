import torch
import triton
import triton.language as tl


@triton.jit
def gqa_gather_kernel(kv_ptr, out_ptr, num_kv_heads, dim: tl.constexpr):
    q_head = tl.program_id(0)
    kv_head = q_head % num_kv_heads
    offs_d = tl.arange(0, dim)
    kv = tl.load(kv_ptr + kv_head * dim + offs_d)
    tl.store(out_ptr + q_head * dim + offs_d, kv)


def gqa_gather(kv: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv_heads, dim = kv.shape
    out = torch.empty(num_q_heads, dim, device=kv.device, dtype=kv.dtype)
    gqa_gather_kernel[(num_q_heads,)](kv, out, num_kv_heads, dim=dim)
    return out
