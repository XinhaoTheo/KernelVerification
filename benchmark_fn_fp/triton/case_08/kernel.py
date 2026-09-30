"""Triton kernel under test: fp5_stochastic_rounding_philox."""
import torch
import triton
import triton.language as tl

STEP = 0.05


@triton.jit
def stochastic_round_kernel(x_ptr, out_ptr, seed, N: tl.constexpr, STEP: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    lower = tl.floor(x / STEP) * STEP
    upper = lower + STEP
    p_up = (x - lower) / STEP
    r = tl.rand(seed, offs)  # real Triton Philox RNG
    out = tl.where(r < p_up, upper, lower)
    tl.store(out_ptr + offs, out)


def stochastic_round_to_grid(x: torch.Tensor, seed: int) -> torch.Tensor:
    N = x.shape[0]
    out = torch.empty_like(x)
    stochastic_round_kernel[(1,)](x, out, seed, N=N, STEP=STEP)
    return out
