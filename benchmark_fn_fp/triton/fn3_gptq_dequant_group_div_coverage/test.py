import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import gptq_matmul  # real, unmodified AutoGPTQ kernel


def build_packed_weight(K, N, bits, device):
    maxq = 2 ** bits - 1
    infeature_per_bits = 32 // bits
    w_int = torch.randint(0, maxq + 1, (K, N), device=device, dtype=torch.int32)
    packed = torch.zeros((K // infeature_per_bits, N), device=device, dtype=torch.int32)
    for row in range(K):
        packed_row = row // infeature_per_bits
        shift = (row % infeature_per_bits) * bits
        packed[packed_row] |= (w_int[row] & maxq) << shift
    return packed, w_int


def run(K, group_size, use_ceil_g_idx, N=32, M=8, bits=4):
    torch.manual_seed(0)
    device = "cuda"
    num_groups = math.ceil(K / group_size) if use_ceil_g_idx else K // group_size
    a = torch.rand(M, K, device=device, dtype=torch.float32)
    packed, w_int = build_packed_weight(K, N, bits, device)
    scales = (torch.arange(num_groups, device=device, dtype=torch.float32) * 0.5 + 1.0).unsqueeze(-1).expand(-1, N).contiguous()
    zeros = torch.zeros((num_groups, N), device=device, dtype=torch.int32)  # zero-point already applied in w_int for simplicity: use zeros=0, kernel adds +1
    zeros_shift_free = torch.zeros_like(zeros)

    g_idx = torch.clamp(torch.arange(K, device=device, dtype=torch.int32) // group_size, max=num_groups - 1)

    out = gptq_matmul(a, packed, scales, zeros_shift_free, g_idx, bits=bits)

    # exact fp32 dequant reference (independent of the kernel, using the SAME g_idx/scales)
    group_of_col = g_idx.long()
    w_dequant = (w_int.float() - 1.0) * scales[group_of_col]
    ref = a @ w_dequant
    return out, ref, g_idx, num_groups


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"

    out_div, ref_div, _, _ = run(K=4096, group_size=128, use_ceil_g_idx=True)
    out_div_bug, _, _, _ = run(K=4096, group_size=128, use_ceil_g_idx=False)
    divisible_pass = torch.allclose(out_div_bug, ref_div, rtol=1e-2, atol=1e-2)

    out_irr, ref_irr, _, _ = run(K=304, group_size=128, use_ceil_g_idx=True)
    out_irr_bug, _, _, _ = run(K=304, group_size=128, use_ceil_g_idx=False)
    irregular_pass = torch.allclose(out_irr_bug, ref_irr, rtol=1e-2, atol=1e-2)

    print(f"real AutoGPTQ kernel, divisible K=4096: allclose={divisible_pass} (bug invisible)")
    print(f"real AutoGPTQ kernel, irregular K=304: allclose={irregular_pass} (bug exposed)")
    print("FN DEMONSTRATED" if divisible_pass and not irregular_pass else "tune constants")
    # What a conventional CI test would conclude: 常规 K=4096（整除），tail 分支走不到
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
