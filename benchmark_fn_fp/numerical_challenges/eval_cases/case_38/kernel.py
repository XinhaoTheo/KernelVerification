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
    order = np.array([98, 70, 9, 33, 119, 58, 66, 36, 32, 118, 104, 94, 105, 7, 89, 102, 125, 10, 12, 61, 126, 18, 111, 117, 51, 64, 62, 75, 16, 3, 26, 56, 54, 101, 120, 88, 71, 78, 97, 22, 6, 63, 41, 87, 86, 85, 127, 35, 43, 112, 80, 81, 28, 8, 45, 84, 79, 99, 115, 40, 4, 11, 93, 24, 48, 74, 25, 114, 44, 121, 77, 106, 27, 20, 92, 1, 47, 37, 42, 49, 5, 109, 30, 96, 34, 15, 116, 50, 83, 60, 67, 76, 13, 14, 122, 53, 91, 59, 110, 73, 113, 31, 107, 69, 100, 68, 23, 82, 72, 38, 52, 17, 95, 55, 103, 39, 90, 123, 2, 65, 0, 124, 29, 21, 19, 108, 57, 46], dtype=np.int64)
    return logits, np.ascontiguousarray(values[order])


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
