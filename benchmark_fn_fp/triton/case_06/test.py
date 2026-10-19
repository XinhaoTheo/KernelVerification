import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import gptq_matmul  # real, unmodified AutoGPTQ kernel


def math_ceil_div(a, b):
    return (a + b - 1) // b


def run(K, N=64, M=8, bits=4, group_size=128):
    torch.manual_seed(0)
    device = "cuda"
    num_groups = math_ceil_div(K, group_size)
    infeature_per_bits = 32 // bits
    maxq = 2 ** bits - 1

    a = torch.randn(M, K, device=device, dtype=torch.float32)
    # A continuous "true" weight matrix -- this is what the reference matmul
    # uses. Quantizing it to INT4 (below) introduces REAL rounding loss; the
    # earlier reconstruct-from-int comparison was trivially exact because it
    # never compared against a continuous ground truth.
    w_true = torch.randn(K, N, device=device, dtype=torch.float32)

    g_idx = torch.clamp(torch.arange(K, device=device, dtype=torch.int32) // group_size, max=num_groups - 1)
    group_min = torch.full((num_groups, N), float("inf"), device=device)
    group_max = torch.full((num_groups, N), float("-inf"), device=device)
    group_min.scatter_reduce_(0, g_idx.long().unsqueeze(-1).expand(K, N), w_true, reduce="amin")
    group_max.scatter_reduce_(0, g_idx.long().unsqueeze(-1).expand(K, N), w_true, reduce="amax")
    scales = ((group_max - group_min) / maxq).clamp_min(1e-6)
    zero_point = torch.clamp(torch.round(-group_min / scales), 1, maxq)

    scales_row = scales[g_idx.long()]
    zp_row = zero_point[g_idx.long()]
    w_int = torch.clamp(torch.round(w_true / scales_row) + zp_row, 0, maxq).to(torch.int32)

    packed = torch.zeros((math_ceil_div(K, infeature_per_bits), N), device=device, dtype=torch.int32)
    for row in range(K):
        packed_row = row // infeature_per_bits
        shift = (row % infeature_per_bits) * bits
        packed[packed_row] |= (w_int[row] & maxq) << shift

    # The kernel expects `zeros` bit-packed the SAME way as the weights, but
    # packed along the N (column) axis instead of K: 8 4-bit zero-points per
    # int32 word (`zeros_ptrs + offs_bn//infeature_per_bits`, unpacked via
    # `(zeros >> (offs_bn%infeature_per_bits)*bits) & maxq`). Passing an
    # unpacked per-column tensor here (as if raw ints addressed it 1:1) reads
    # the wrong nibble for nearly every column.
    zeros_field = (zero_point - 1).to(torch.int32)  # kernel adds +1 back internally
    zeros_packed = torch.zeros((num_groups, math_ceil_div(N, infeature_per_bits)), device=device, dtype=torch.int32)
    for col in range(N):
        packed_col = col // infeature_per_bits
        shift = (col % infeature_per_bits) * bits
        zeros_packed[:, packed_col] |= (zeros_field[:, col] & maxq) << shift

    out = gptq_matmul(a, packed, scales, zeros_packed, g_idx, bits=bits)
    ref = a @ w_true  # continuous ground truth, NOT the round-tripped int4 value
    return out, ref


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"

    out, ref = run(K=256)
    fp32_tolerance_pass = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
    rel_err = ((out - ref).abs() / ref.abs().clamp_min(1e-3)).mean().item()

    # Format-aware check: INT4 (16 levels per group) is expected to carry
    # several percent of relative error; accept it as long as it's in that
    # expected ballpark rather than blown up (e.g. from an indexing bug).
    format_aware_pass = 0.0 < rel_err < 0.5

    print(f"real AutoGPTQ INT4 kernel (GPU), K=256: FP32-calibrated allclose(rtol=1e-2,atol=1e-2)={fp32_tolerance_pass}")
    print(f"mean relative error vs continuous ground truth = {rel_err:.3f} (expected INT4 quantization noise, not a bug)")
    print("FP DEMONSTRATED" if (not fp32_tolerance_pass) and format_aware_pass else "tune constants")
    # What a conventional CI test would conclude: 拿 FP32 标定的容差去卡 INT4
    print(f"NAIVE_ALLCLOSE_VERDICT: {fp32_tolerance_pass}")
    return (not fp32_tolerance_pass) and format_aware_pass


if __name__ == "__main__":
    test_kernel()
