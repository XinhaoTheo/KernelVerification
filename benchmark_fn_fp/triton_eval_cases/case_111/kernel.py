"""Standalone RMSNorm backward kernels adapted from Liger Kernel.

Copyright 2024 LinkedIn Corporation. See LICENSE for license and attribution.
2026 adaptation: standalone CUDA interface, FP16 activations and FP32 weight
accumulation buffer. The public contract defines the supported casting policy.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _row_backward(DY, DX, X, W, RSTD, PARTIAL, M: tl.constexpr,
                  N: tl.constexpr, ROWS: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, B)
    mask = cols < N
    weight = tl.load(W + cols, mask=mask, other=0.0)
    dw = tl.zeros((B,), tl.float32)
    start = pid * ROWS
    stop = tl.minimum(start + ROWS, M)
    for row in range(start, stop):
        dy = tl.load(DY + row * N + cols, mask=mask, other=0.0)
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RSTD + row)
        m = (dy * weight).to(tl.float32)
        dx = r * m
        dx += r * (-(1.0 / N) * r * r * tl.sum(m * x, 0) * x)
        dw += dy * (x * r).to(tl.float16)
        tl.store(DX + row * N + cols, dx, mask=mask)
    tl.store(PARTIAL + pid * N + cols, dw, mask=mask)


@triton.jit
def _block_backward(DY, DX, X, W, RSTD, PARTIAL, M: tl.constexpr,
                    N: tl.constexpr, B: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    programs = tl.num_programs(0)
    cols = tl.arange(0, B)
    col_mask = cols < N
    weight = tl.load(W + cols, mask=col_mask, other=0.0)
    dw = tl.zeros((B,), tl.float32)
    for start in range(pid * BR, M, programs * BR):
        rows = start + tl.arange(0, BR)
        mask = (rows[:, None] < M) & col_mask[None, :]
        dy = tl.load(DY + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)
        x = tl.load(X + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RSTD + rows, mask=rows < M, other=0.0)
        m = (dy * weight[None, :]).to(tl.float32)
        dx = r[:, None] * m
        dx += r[:, None] * (
            -(1.0 / N) * (r * r * tl.sum(m * x, 1))[:, None] * x
        )
        dw += tl.sum((dy * (x * r[:, None]).to(tl.float16)).to(tl.float32), 0)
        tl.store(DX + rows[:, None] * N + cols[None, :], dx, mask=mask)
    tl.store(PARTIAL + pid * N + cols, dw, mask=col_mask)


def run(x, weight, dy, rstd):
    """Return (dx: FP16[M,N], dw: FP32[N]); preserve all inputs."""
    x, weight, dy, rstd = (t.resolve_neg() for t in (x, weight, dy, rstd))
    m, n = x.shape
    assert x.is_cuda and x.is_contiguous() and dy.is_contiguous()
    assert weight.is_contiguous() and rstd.is_contiguous()
    assert x.dtype == dy.dtype == weight.dtype == torch.float16
    assert rstd.dtype == torch.float32
    assert dy.shape == x.shape and weight.shape == (n,) and rstd.shape == (m,)
    assert x.device == weight.device == dy.device == rstd.device
    assert 1 <= m <= 4096 and 16 <= n <= 512
    block = triton.next_power_of_2(n)
    programs = min(torch.cuda.get_device_properties(x.device).multi_processor_count, m)
    partial = torch.empty((programs, n), dtype=torch.float32, device=x.device)
    dx = torch.empty_like(dy)
    if block > 256 or m < 512:
        _row_backward[(programs,)](
            dy, dx, x, weight, rstd, partial, m, n,
            ROWS=triton.cdiv(m, programs), B=block,
            num_warps=4, enable_fp_fusion=False,
        )
    else:
        _block_backward[(programs,)](
            dy, dx, x, weight, rstd, partial, m, n,
            B=block, BR=16, num_warps=4, enable_fp_fusion=False,
        )
    dw = partial.sum(dim=0)
    return dx, dw


def reference(x, weight, dy, rstd):
    """Independent tensor reference for the public mixed-precision contract.

    FP64 evaluates the analytic dx formula and accumulates the individually
    rounded weight-gradient terms. This function can also run on CPU.
    """
    x64 = x.to(torch.float64)
    r64 = rstd.to(torch.float64)[:, None]
    m64 = (dy * weight).to(torch.float64)
    dx64 = r64 * (m64 - x64 * r64.square() * (m64 * x64).mean(dim=1, keepdim=True))
    h = (x.float() * rstd[:, None]).to(torch.float16)
    terms64 = (dy * h).to(torch.float64)
    return dx64, terms64.sum(dim=0)


def error_ratios(outputs, inputs):
    """A value <= 1 for each output satisfies the numerical contract."""
    dx, dw = outputs
    x, weight, dy, rstd = inputs
    ref_dx, ref_dw = reference(*inputs)
    h = (x.float() * rstd[:, None]).to(torch.float16)
    abs_terms = (dy * h).to(torch.float64).abs().sum(dim=0)
    dx_limit = 0.002 + 0.002 * ref_dx.abs()
    dw_limit = 0.00001 + 0.00001 * abs_terms
    return {
        "dx": float(((dx.double() - ref_dx).abs() / dx_limit).max().item()),
        "dw": float(((dw.double() - ref_dw).abs() / dw_limit).max().item()),
    }


def make_inputs(device="cuda", rows=768, cols=128, seed=0):
    """An ordinary example, not a restriction of the supported input domain."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (2 * torch.rand((rows, cols), generator=g) - 1).half()
    weight = (0.5 + torch.rand((cols,), generator=g)).half()
    dy = (2 * torch.rand((rows, cols), generator=g) - 1).half()
    rstd = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    return tuple(t.to(device) for t in (x, weight, dy, rstd))
