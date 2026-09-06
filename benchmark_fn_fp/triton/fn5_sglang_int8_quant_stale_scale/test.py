import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import quant_dequant_int8 as quant_dequant_buggy

import triton
import triton.language as tl


@triton.jit
def _per_token_quant_int8_ref(x_ptr, xq_ptr, scale_ptr, stride_x, stride_xq, N, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)  # real: current-row max
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.clamp(tl.extra.cuda.libdevice.round(x_q), -127, 127).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def quant_dequant_int8_reference(x):
    M, N = x.shape
    x_q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    _per_token_quant_int8_ref[(M,)](x, x_q, scales, x.stride(0), x_q.stride(0), N, BLOCK=BLOCK)
    return x_q.float() * scales.unsqueeze(-1)


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    torch.manual_seed(0)

    # Mild: construct the row so its true absmax is EXACTLY the stale constant
    # (4.0), by directly setting one entry to that float value rather than
    # normalizing by division (which can leave the max a few ULPs off from
    # 4.0 and flip a handful of int8 rounding decisions at the bin edge).
    x_mild = torch.randn(4, 1024, device=device) * 0.5
    x_mild[:, 0] = 4.0
    ref_mild = quant_dequant_int8_reference(x_mild)
    cand_mild = quant_dequant_buggy(x_mild)
    mild_pass = torch.allclose(cand_mild, ref_mild, rtol=1e-2, atol=1e-2)

    # Adversarial: one row has a real outlier far beyond the stale scale.
    x_out = torch.randn(4, 1024, device=device)
    x_out[0, 0] = 50.0
    ref_out = quant_dequant_int8_reference(x_out)
    cand_out = quant_dequant_buggy(x_out)
    outlier_pass = torch.allclose(cand_out, ref_out, rtol=1e-2, atol=1e-2)

    print(f"mild (true absmax == stale scale): allclose={mild_pass}")
    print(f"outlier row: allclose={outlier_pass}, reconstructed={cand_out[0,0].item():.2f} vs true {x_out[0,0].item():.2f}")
    print("FN DEMONSTRATED" if mild_pass and not outlier_pass else "tune constants")
    # What a conventional CI test would conclude: 合成高斯输入，没有 outlier
    print(f"NAIVE_ALLCLOSE_VERDICT: {mild_pass}")
    return mild_pass and not outlier_pass


if __name__ == "__main__":
    test_kernel()
