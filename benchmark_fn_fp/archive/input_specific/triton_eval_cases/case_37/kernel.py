import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(X, A, B, Y, K: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, K)
    x = tl.load(X + j)
    a = tl.load(A + row * K + j)
    b = tl.load(B + row * K + j)
    sa = tl.max(tl.abs(a), 0) / 7.0
    sb = tl.max(tl.abs(b), 0) / 7.0
    qa = tl.minimum(7.0, tl.maximum(-7.0, tl.floor(a / sa + 0.5)))
    qb = tl.minimum(7.0, tl.maximum(-7.0, tl.floor(b / sb + 0.5)))
    ya = tl.sum((qa * sa) * x, 0)
    yb = tl.sum((qb * sb) * x, 0)
    tl.store(Y + row, ya + yb)

def run(x, a, b):
    out = torch.empty(a.shape[0], device=x.device, dtype=torch.float32)
    _kernel[(a.shape[0],)](x, a, b, out, a.shape[1], enable_fp_fusion=False)
    return out

SEED = 921000
PERMUTATION = [5, 20, 47, 51, 15, 57, 30, 41, 55, 13, 18, 49, 52, 24, 45, 1, 12, 37, 53, 25, 38, 31, 28, 6, 9, 29, 40, 34, 60, 17, 56, 35, 63, 39, 11, 50, 58, 10, 43, 23, 36, 62, 21, 2, 27, 0, 26, 16, 48, 42, 33, 59, 7, 54, 32, 44, 3, 4, 61, 46, 8, 14, 22, 19]

def make_inputs(device="cuda"):
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(SEED))
    x = rng.standard_normal(128)
    x /= np.linalg.norm(x)
    # Construct two dense projections with similar signal magnitudes.
    matrices = []
    for _ in range(2):
        w = rng.standard_normal((64, 128))
        target = 0.5 + 0.02 * rng.standard_normal(64)
        projection = np.sum(w * x[None, :], axis=1, dtype=np.float64)
        w += ((target - projection) / np.sum(x * x))[:, None] * x[None, :]
        matrices.append(w.astype(np.float32))
    arrays = [x.astype(np.float32), matrices[0], matrices[1][PERMUTATION].copy()]
    return tuple(torch.from_numpy(a.copy()).to(device) for a in arrays)
