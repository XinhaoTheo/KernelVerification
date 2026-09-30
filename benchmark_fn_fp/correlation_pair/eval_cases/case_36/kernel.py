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
PERMUTATION = [4, 1, 2, 53, 44, 13, 11, 56, 46, 57, 33, 8, 25, 35, 39, 20, 36, 14, 51, 52, 40, 29, 23, 26, 3, 31, 38, 16, 21, 62, 41, 24, 61, 45, 30, 59, 19, 42, 27, 28, 12, 17, 60, 47, 43, 54, 6, 34, 22, 10, 18, 50, 32, 0, 7, 15, 9, 5, 63, 55, 49, 37, 48, 58]

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
