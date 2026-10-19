import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import rsqrt
except ModuleNotFoundError:
    from triton.language.extra.cuda.libdevice import rsqrt


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr, Y_row_stride, X_ptr, X_row_stride, RSTD_ptr, RSTD_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    X_row = tl.load(X_ptr + row_idx * X_row_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    tl.store(RSTD_ptr + row_idx * RSTD_row_stride, rstd)
    Y_row = X_row * rstd
    tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=mask)


def rms_norm_forward(X: torch.Tensor, eps: float) -> torch.Tensor:
    num_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    X32 = X.float()
    Y = torch.empty(num_rows, n_cols, device=X.device, dtype=torch.float32)
    RSTD = torch.empty(num_rows, dtype=torch.float32, device=X.device)
    _rms_norm_forward_kernel[(num_rows,)](Y, Y.stride(0), X32, X32.stride(0), RSTD, RSTD.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return Y
