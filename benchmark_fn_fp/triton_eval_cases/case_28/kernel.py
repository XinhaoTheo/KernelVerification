import torch
import triton
import triton.language as tl


@triton.jit
def _quant_kernel(X, OUT, stride, N, PCTL_LEVELS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < N
    x = tl.load(X + row * stride + cols, mask=valid, other=0.0)

    ax = tl.abs(x)
    absmax = tl.max(ax, axis=0)
    # Calibrate on the bulk of the row rather than on its single largest value.
    thresh = absmax
    for _ in range(PCTL_LEVELS):
        below = tl.where(ax < thresh, ax, 0.0)
        thresh = tl.max(below, axis=0)

    scale = tl.where(thresh == 0.0, 1.0, thresh / 127.0)
    q = tl.floor(x / scale + 0.5)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(OUT + row * stride + cols, q * scale, mask=valid)


def quant_dequant(x, pctl_levels: int = 2):
    # Symmetric INT8 quantize-dequantize with a per-row calibrated scale.
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    _quant_kernel[(n_rows,)](x, out, x.stride(0), n_cols,
                             PCTL_LEVELS=pctl_levels,
                             BLOCK=triton.next_power_of_2(n_cols))
    return out
