import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct(Coefficients, Offset, Output, N: tl.constexpr,
                 MODES: tl.constexpr, RETAINED: tl.constexpr):
    n = tl.arange(0, N)
    output = tl.full((N,), 0.0, tl.float32) + tl.load(Offset)
    for k in tl.static_range(1, RETAINED + 1):
        a = tl.load(Coefficients + k - 1)
        b = tl.load(Coefficients + MODES + k - 1)
        angle = n.to(tl.float32) * (2.0 * 3.141592653589793 * k / N)
        output = output + a * tl.cos(angle)
        output = output + b * tl.sin(angle)
    tl.store(Output + n, output)


def run(coefficients, offset):
    output = torch.empty((64,), device=coefficients.device, dtype=torch.float32)
    _reconstruct[(1,)](coefficients, offset, output, N=64, MODES=16,
                       RETAINED=6, num_warps=2, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(811158))
    coefficients = rng.normal(0.0, 0.04, (2, 16)).astype(np.float32)
    coefficients[:, :6] = rng.normal(0.0, 0.15, (2, 6)).astype(np.float32)
    coefficients[0, 0] = np.float32(coefficients[0, 0] + np.float32(1.0))
    offset = np.asarray([0.25], dtype=np.float32)
    return coefficients, offset


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
