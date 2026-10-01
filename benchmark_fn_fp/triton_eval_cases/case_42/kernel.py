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
PERMUTATION = [123, 54, 34, 37, 17, 97, 46, 110, 36, 107, 32, 114, 14, 86, 40, 33, 122, 103, 70, 121, 63, 80, 62, 65, 72, 68, 55, 105, 113, 53, 7, 108, 59, 93, 58, 125, 89, 94, 18, 87, 30, 82, 56, 3, 119, 96, 127, 118, 20, 22, 24, 60, 117, 35, 16, 85, 41, 76, 81, 48, 0, 5, 101, 26, 44, 11, 51, 43, 104, 71, 9, 78, 39, 84, 90, 47, 45, 115, 57, 98, 66, 79, 77, 52, 49, 2, 91, 116, 4, 100, 19, 75, 69, 112, 120, 21, 88, 23, 109, 15, 27, 126, 28, 31, 6, 29, 61, 38, 92, 102, 73, 83, 95, 42, 67, 64, 99, 25, 106, 1, 8, 50, 74, 124, 12, 111, 10, 13]

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
