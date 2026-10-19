import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import gptq_matmul


def build_packed_weight(K, N, bits, device):
    maxq = 2 ** bits - 1
    infeature_per_bits = 32 // bits
    w_int = torch.randint(0, maxq + 1, (K, N), device=device, dtype=torch.int32)
    packed = torch.zeros((K // infeature_per_bits, N), device=device, dtype=torch.int32)
    for row in range(K):
        packed[row // infeature_per_bits] |= (w_int[row] & maxq) << ((row % infeature_per_bits) * bits)
    return packed, w_int


def run(K, group_size, N=32, M=8, bits=4):
    """Call the kernel, and build the reference from the CONTRACT's group table.

    The contract says the trailing partial group is a group of its own, so the
    reference uses ceil(K / group_size) groups. The wrapper under test derives
    its own table from group_size; whether it agrees is the whole question.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_groups = math.ceil(K / group_size)

    a = torch.rand(M, K, device=device, dtype=torch.float32)
    packed, w_int = build_packed_weight(K, N, bits, device)
    # Distinct scale per group, so reading the wrong group's row is visible.
    scales = ((torch.arange(num_groups, device=device, dtype=torch.float32) * 0.5 + 1.0)
              .unsqueeze(-1).expand(-1, N).contiguous())
    zeros = torch.zeros((num_groups, N), device=device, dtype=torch.int32)

    out = gptq_matmul(a, packed, scales, zeros, group_size, bits=bits)

    group_of_col = (torch.arange(K, device=device) // group_size).long()
    w_dequant = (w_int.float() - 1.0) * scales[group_of_col]
    ref = a @ w_dequant
    return out, ref


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"

    out_div, ref_div = run(K=4096, group_size=128)
    divisible_pass = torch.allclose(out_div, ref_div, rtol=1e-2, atol=1e-2)

    out_irr, ref_irr = run(K=304, group_size=128)
    irregular_pass = torch.allclose(out_irr, ref_irr, rtol=1e-2, atol=1e-2)
    max_err = (out_irr - ref_irr).abs().max().item()
    ref_absmax = ref_irr.abs().max().item()

    print(f"divisible K=4096, group_size=128: allclose={divisible_pass} (32 full groups, floor == ceil)")
    print(f"irregular K=304, group_size=128: allclose={irregular_pass} "
          f"max_abs_err={max_err:.4f} ref_absmax={ref_absmax:.4f} "
          f"(columns 256..303 are the short final group)")
    print("FN DEMONSTRATED" if divisible_pass and not irregular_pass else "tune constants")
    # A conventional CI test picks the convenient divisible shape and sees nothing.
    print(f"NAIVE_ALLCLOSE_VERDICT: {divisible_pass}")
    return divisible_pass and not irregular_pass


if __name__ == "__main__":
    test_kernel()
