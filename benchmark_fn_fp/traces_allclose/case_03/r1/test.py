import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import rms_norm_forward  # real, unmodified kernel -- used for BOTH sides


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    eps = 1e-6

    x_fp32 = torch.full((1, 16), 3e-4, device=device)
    x_bf16_roundtrip = x_fp32.bfloat16().float()  # ordinary BF16 rounding of the input

    out_fp32 = rms_norm_forward(x_fp32, eps)
    out_bf16 = rms_norm_forward(x_bf16_roundtrip, eps)

    abs_diff = (out_fp32 - out_bf16).abs().max().item()
    relative_only_pass = abs_diff <= 1e-2 * out_fp32.abs().max().item()
    combined_pass = torch.allclose(out_fp32, out_bf16, rtol=1e-2, atol=1e-2)

    print(f"real Liger RMSNorm kernel (GPU): abs diff = {abs_diff:.3e}, output magnitude ~ {out_fp32.abs().max().item():.3f}")
    print(f"relative-error-only check = {relative_only_pass}")
    print(f"torch.allclose (atol+rtol combined) = {combined_pass} (expected True)")
    print("FP DEMONSTRATED" if combined_pass else "tune constants")
    # What a conventional CI test would conclude: 只用相对误差，近零处该指标失效
    print(f"NAIVE_ALLCLOSE_VERDICT: {relative_only_pass}")
    return combined_pass


if __name__ == "__main__":
    test_kernel()
