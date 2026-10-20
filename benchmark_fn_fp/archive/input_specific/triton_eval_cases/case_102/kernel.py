import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(Logits, Values, Output, N: tl.constexpr, D: tl.constexpr):
    i = tl.arange(0, N)
    j = tl.arange(0, D)
    logits = tl.load(Logits + i).to(tl.float32)
    scale = tl.max(tl.abs(logits), axis=0) / 7.0
    codes = tl.minimum(tl.maximum(tl.floor(logits / scale + 0.5), -7.0), 7.0)
    rounded = codes * scale
    weights = tl.exp(rounded - tl.max(rounded, axis=0))
    probabilities = weights / tl.sum(weights, axis=0)
    values = tl.load(Values + i[:, None] * D + j[None, :]).to(tl.float32)
    result = tl.sum(probabilities[:, None] * values, axis=0)
    tl.store(Output + j, result)


def run(logits, values):
    output = torch.empty((16,), device=logits.device, dtype=torch.float32)
    _attention_kernel[(1,)](logits, values, output, N=128, D=16, num_warps=4)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(918233))
    logits = rng.normal(0.0, 1.2, 128).astype(np.float32)
    values = (1.0 + rng.normal(0.0, 1.0, (128, 16))).astype(np.float32)
    order = np.array([91, 7, 75, 33, 78, 39, 30, 46, 107, 57, 47, 15, 40, 17, 81, 123, 0, 54, 13, 58, 90, 84, 35, 115, 24, 72, 65, 31, 82, 86, 25, 83, 99, 77, 21, 70, 71, 73, 56, 32, 60, 100, 19, 50, 112, 127, 98, 120, 51, 26, 95, 69, 80, 36, 111, 92, 124, 108, 38, 11, 126, 23, 104, 28, 8, 97, 61, 67, 45, 5, 116, 29, 113, 119, 114, 34, 22, 103, 59, 27, 117, 62, 102, 88, 14, 63, 10, 3, 37, 76, 4, 66, 106, 101, 41, 20, 49, 12, 9, 85, 55, 48, 18, 44, 2, 6, 94, 89, 43, 16, 87, 42, 105, 64, 96, 1, 118, 68, 52, 121, 74, 53, 122, 109, 79, 125, 110, 93], dtype=np.int64)
    return logits, np.ascontiguousarray(values[order])


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
