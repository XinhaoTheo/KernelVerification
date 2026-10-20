import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _solve_kernel(Matrix, RHS, Output, N: tl.constexpr, STEPS: tl.constexpr):
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    matrix = tl.load(Matrix + i[:, None] * N + j[None, :]).to(tl.float32)
    rhs = tl.load(RHS + i).to(tl.float32)
    state = tl.full((N,), 0.0, tl.float32)
    for step in range(STEPS):
        residual = rhs - tl.sum(matrix * state[None, :], axis=1)
        state = state + residual
    tl.store(Output + i, state)


def run(matrix, rhs):
    output = torch.empty((16,), device=matrix.device, dtype=torch.float32)
    _solve_kernel[(1,)](matrix, rhs, output, N=16, STEPS=64,
                       num_warps=4, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    orthogonal = np.array([[1.0]], dtype=np.float64)
    while len(orthogonal) < 16:
        orthogonal = np.block([[orthogonal, orthogonal], [orthogonal, -orthogonal]])
    orthogonal /= 4.0
    eigenvalues = np.array([0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
                            0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75],
                           dtype=np.float64)
    matrix = np.sum(orthogonal[:, :, None] * eigenvalues[None, :, None]
                    * orthogonal.T[None, :, :], axis=1).astype(np.float32)
    rng = np.random.Generator(np.random.PCG64(701010))
    rhs = rng.normal(0.0, 1.0, 16).astype(np.float32)
    norm = np.sqrt(np.sum(rhs.astype(np.float64) ** 2, dtype=np.float64))
    rhs = (rhs.astype(np.float64) / norm).astype(np.float32)
    return matrix, rhs


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
