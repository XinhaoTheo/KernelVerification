import torch
import triton
import triton.language as tl


@triton.jit
def rotary_kernel(
    OUT, X, COS, SIN,
    seqlen, nheads, rotary_dim_half,
    stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    INTERLEAVED: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_head = tl.program_id(0)
    pid_m = tl.program_id(1)
    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk_half = tl.arange(0, BLOCK_K // 2)

    COS = COS + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm[:, None] * rotary_dim_half + rk_half[None, :])
    mask_cs = (rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)

    if not INTERLEAVED:
        X_ptrs = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk_half[None, None, :] * stride_x_headdim)
        OUT_ptrs = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk_half[None, None, :] * stride_out_headdim)
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk_half[None, None, :] < rotary_dim_half)
        x0 = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X_ptrs + rotary_dim_half * stride_x_headdim, mask=mask, other=0.0).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT_ptrs, o0, mask=mask)
        tl.store(OUT_ptrs + rotary_dim_half * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X_ptrs = X + (rh[:, None, None] * stride_x_nheads + rm[None, :, None] * stride_x_seqlen + rk[None, None, :] * stride_x_headdim)
        OUT_ptrs = OUT + (rh[:, None, None] * stride_out_nheads + rm[None, :, None] * stride_out_seqlen + rk[None, None, :] * stride_out_headdim)
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk[None, None, :] < 2 * rotary_dim_half)
        x = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT_ptrs, o, mask=mask)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """x: (seqlen, nheads, headdim). cos/sin: (seqlen, headdim//2)."""
    seqlen, nheads, headdim = x.shape
    rotary_dim_half = headdim // 2
    out = torch.empty_like(x)
    BLOCK_K = triton.next_power_of_2(headdim)
    grid = (nheads, seqlen)
    rotary_kernel[grid](
        out, x, cos, sin,
        seqlen, nheads, rotary_dim_half,
        out.stride(0), out.stride(1), out.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        INTERLEAVED=interleaved, BLOCK_H=1, BLOCK_M=1, BLOCK_K=BLOCK_K,
    )
    return out
