import torch
import triton
import triton.language as tl

_STALE_ABSMAX = 4.0  # simulates an EMA scale that hasn't caught up to the current batch


@triton.jit
def _per_token_quant_int8(
    x_ptr, xq_ptr, scale_ptr,
    stride_x, stride_xq, N,
    STALE_ABSMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = STALE_ABSMAX
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.clamp(tl.extra.cuda.libdevice.round(x_q), -127, 127).to(tl.int8)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def quant_dequant_int8(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    x_q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    _per_token_quant_int8[(M,)](x, x_q, scales, x.stride(0), x_q.stride(0), N, STALE_ABSMAX=_STALE_ABSMAX, BLOCK=BLOCK)
    return x_q.float() * scales.unsqueeze(-1)
