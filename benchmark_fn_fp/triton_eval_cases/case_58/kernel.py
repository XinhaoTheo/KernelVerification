import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _logdet_kernel(Matrix, Output, N: tl.constexpr):
    rows = tl.arange(0, N)
    columns = tl.arange(0, N)
    matrix = tl.load(Matrix + rows[:, None] * N + columns[None, :]).to(tl.float32)
    output = tl.full((), 0.0, tl.float32)
    for k in tl.static_range(0, N):
        pivot = tl.sum(tl.sum(tl.where((rows[:, None] == k) &
                                      (columns[None, :] == k), matrix, 0.0), 0), 0)
        column = tl.sum(tl.where(columns[None, :] == k, matrix, 0.0), 1)
        pivot_row = tl.sum(tl.where(rows[:, None] == k, matrix, 0.0), 0)
        multiplier = tl.div_rn(column, pivot)
        product = multiplier[:, None] * pivot_row[None, :]
        updated = matrix - product
        matrix = tl.where((rows[:, None] > k) & (columns[None, :] > k),
                          updated, matrix)
        output = output + tl.log(pivot)
    tl.store(Output, output)


def run(matrix):
    output = torch.empty((1,), device=matrix.device, dtype=torch.float32)
    _logdet_kernel[(1,)](matrix, output, N=8, num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(98339))
    factor = rng.integers(-8, 9, (8, 7), dtype=np.int64)
    gram = np.sum(factor[:, None, :] * factor[None, :, :], axis=2, dtype=np.int64)
    matrix = (gram.astype(np.float64) + (1.0 / 1024.0) * np.eye(8)).astype(np.float32)
    return (matrix,)


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
