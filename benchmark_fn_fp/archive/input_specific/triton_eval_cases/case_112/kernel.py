import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(X, Y, K: tl.constexpr, R: tl.constexpr):
    rows = tl.arange(0, R)
    acc = tl.full((R,), 0, tl.float32)
    for j in range(K):
        value = tl.load(X + rows * K + j)
        acc = acc + value
    tl.store(Y + rows, acc)

def run(values):
    out = torch.empty((values.shape[0],), dtype=torch.float32, device=values.device)
    _kernel[(1,)](values, out, values.shape[1], values.shape[0], enable_fp_fusion=False)
    return out

SEED = 730119
PERMUTATION = [40, 110, 94, 84, 77, 58, 57, 120, 82, 13, 85, 86, 113, 106, 39, 102, 19, 18, 112, 26, 71, 122, 61, 103, 46, 91, 79, 125, 44, 38, 47, 28, 50, 30, 97, 52, 53, 109, 119, 35, 1, 107, 43, 114, 117, 60, 74, 33, 25, 88, 78, 59, 81, 93, 27, 36, 108, 90, 23, 126, 95, 76, 31, 124, 98, 73, 37, 29, 32, 10, 48, 51, 17, 121, 100, 6, 22, 49, 11, 20, 34, 69, 115, 4, 68, 56, 5, 21, 66, 65, 92, 3, 15, 0, 118, 67, 55, 83, 105, 116, 99, 101, 42, 16, 123, 64, 104, 41, 87, 9, 62, 8, 12, 127, 14, 24, 75, 70, 80, 7, 54, 45, 96, 2, 63, 89, 111, 72]

def make_inputs_numpy():
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(SEED))
    w = rng.integers(1, 33, size=(64, 32)).astype(np.float32) * np.float32(1048576)
    small = rng.integers(1, 4, size=(64, 64)).astype(np.float32) * np.float32(0.25)
    values = np.concatenate([w, -w, small], axis=1)
    order = rng.permutation(128)
    values = values[:, order][:, PERMUTATION].copy()
    return (values,)

def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(a.copy()).to(device) for a in make_inputs_numpy())
