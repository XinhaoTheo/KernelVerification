"""Triton kernel under test: fn10_int8_group_quant_short_final_group."""
import torch
import triton
import triton.language as tl


@triton.jit
def _group_quant_kernel(X, OUT, stride_xm, stride_om, n_cols, GROUP: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = X + row * stride_xm
    o_row = OUT + row * stride_om
    n_groups = n_cols // GROUP

    for g in range(n_groups):
        cols = g * GROUP + tl.arange(0, BLOCK)
        in_group = tl.arange(0, BLOCK) < GROUP
        x = tl.load(x_row + cols, mask=in_group & (cols < n_cols), other=0.0)
        absmax = tl.max(tl.abs(x), axis=0)
        scale = tl.where(absmax == 0.0, 1.0, absmax / 127.0)
        q = tl.floor(x / scale + 0.5).to(tl.int8)
        deq = q.to(tl.float32) * scale
        tl.store(o_row + cols, deq, mask=in_group & (cols < n_cols))


def group_quant_dequant(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Symmetric INT8 quantize-dequantize with one scale per group of columns."""
    n_rows, n_cols = x.shape
    out = torch.zeros_like(x)
    _group_quant_kernel[(n_rows,)](x, out, x.stride(0), out.stride(0), n_cols,
                                   GROUP=group_size, BLOCK=group_size)
    return out
