import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import rms_norm_forward as rms_norm_buggy

import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import rsqrt
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import rsqrt


@triton.jit
def _rms_norm_forward_kernel_ref(
    Y_ptr, Y_row_stride, X_ptr, X_row_stride, RSTD_ptr, RSTD_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0)
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)  # real kernel: eps inside the sqrt
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)
    Y_row = X_row * rstd
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def rms_norm_forward_reference(X, eps):
    num_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    Y = torch.empty_like(X)
    RSTD = torch.empty(num_rows, dtype=torch.float32, device=X.device)
    _rms_norm_forward_kernel_ref[(num_rows,)](Y, Y.stride(0), X, X.stride(0), RSTD, RSTD.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return Y


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    eps = 1e-6

    x_normal = torch.randn(8, 16, device=device)
    ref_normal = rms_norm_forward_reference(x_normal, eps)
    cand_normal = rms_norm_buggy(x_normal, eps)
    normal_pass = torch.allclose(cand_normal, ref_normal, rtol=1e-2, atol=1e-2)

    x_tiny = torch.full((1, 16), 2e-8, device=device)
    ref_tiny = rms_norm_forward_reference(x_tiny, eps)
    cand_tiny = rms_norm_buggy(x_tiny, eps)
    tiny_pass = torch.allclose(cand_tiny, ref_tiny, rtol=1e-2, atol=1e-2)

    print(f"normal rows: allclose={normal_pass}")
    print(f"near-zero row: allclose={tiny_pass}, ref={ref_tiny.abs().max().item():.3e}, cand={cand_tiny.abs().max().item():.3e}")
    print("FN DEMONSTRATED" if normal_pass and not tiny_pass else "tune constants")
    # What a conventional CI test would conclude: 高斯行，均方永远不接近零
    print(f"NAIVE_ALLCLOSE_VERDICT: {normal_pass}")
    return normal_pass and not tiny_pass


if __name__ == "__main__":
    test_kernel()
