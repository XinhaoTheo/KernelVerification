import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import gqa_gather  # buggy: modulo instead of floor-division

import triton
import triton.language as tl


@triton.jit
def _gqa_gather_kernel_ref(kv_ptr, out_ptr, n_rep, dim: tl.constexpr):
    q_head = tl.program_id(0)
    kv_head = q_head // n_rep  # real: floor-division groups consecutive query heads
    offs_d = tl.arange(0, dim)
    kv = tl.load(kv_ptr + kv_head * dim + offs_d)
    tl.store(out_ptr + q_head * dim + offs_d, kv)


def gqa_gather_reference(kv, num_q_heads):
    num_kv_heads, dim = kv.shape
    n_rep = num_q_heads // num_kv_heads
    out = torch.empty(num_q_heads, dim, device=kv.device, dtype=kv.dtype)
    _gqa_gather_kernel_ref[(num_q_heads,)](kv, out, n_rep, dim=dim)
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    num_q_heads, num_kv_heads, dim = 8, 2, 4

    # (a) homogeneous KV data: every head is the same base vector -> bug invisible
    base = torch.randn(1, dim, device=device)
    kv_homogeneous = base.expand(num_kv_heads, dim).clone().contiguous()
    ref_h = gqa_gather_reference(kv_homogeneous, num_q_heads)
    cand_h = gqa_gather(kv_homogeneous, num_q_heads)
    homogeneous_pass = torch.allclose(cand_h, ref_h, rtol=1e-2, atol=1e-2)

    # (b) distinguishable KV data: bug exposed
    kv_distinct = torch.stack([torch.full((dim,), float(i * 100), device=device) for i in range(num_kv_heads)])
    ref_d = gqa_gather_reference(kv_distinct, num_q_heads)
    cand_d = gqa_gather(kv_distinct, num_q_heads)
    distinct_pass = torch.allclose(cand_d, ref_d, rtol=1e-2, atol=1e-2)

    print(f"homogeneous KV data: allclose={homogeneous_pass} (bug hidden)")
    print(f"distinguishable KV data: allclose={distinct_pass} (bug exposed)")
    print("FN DEMONSTRATED" if homogeneous_pass and not distinct_pass else "tune constants")
    # What a conventional CI test would conclude: 偷懒的同质 KV 数据，路由错也看不出
    print(f"NAIVE_ALLCLOSE_VERDICT: {homogeneous_pass}")
    return homogeneous_pass and not distinct_pass


if __name__ == "__main__":
    test_kernel()
