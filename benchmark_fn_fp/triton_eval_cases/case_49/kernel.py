import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _polynomial_kernel(Coefficients, Points, Output, N: tl.constexpr,
                       DEGREE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    mask = row < N
    point = tl.load(Points + row, mask=mask, other=0.0).to(tl.float32)
    result = tl.load(Coefficients + row * (DEGREE + 1) + DEGREE,
                     mask=mask, other=0.0).to(tl.float32)
    for step in tl.static_range(0, DEGREE):
        k = DEGREE - 1 - step
        coefficient = tl.load(Coefficients + row * (DEGREE + 1) + k,
                              mask=mask, other=0.0).to(tl.float32)
        product = result * point
        result = product + coefficient
    tl.store(Output + row, result, mask=mask)


def run(coefficients, points):
    output = torch.empty((8,), device=coefficients.device, dtype=torch.float32)
    _polynomial_kernel[(1,)](coefficients, points, output, N=8, DEGREE=48,
                             BLOCK=32, num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(501901))
    coefficients = rng.normal(0.0, 1.0, (8, 49)).astype(np.float32)
    anchor = np.float32(1.015625)
    powers = float(anchor) ** np.arange(1, 49)
    coefficients[:, 0] = (
        -np.sum(coefficients[:, 1:].astype(np.float64) * powers[None, :], axis=1)
        + 0.003
    ).astype(np.float32)
    points = (float(anchor) + rng.normal(0.0, 0.00004, 8)).astype(np.float32)
    return coefficients, points


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
