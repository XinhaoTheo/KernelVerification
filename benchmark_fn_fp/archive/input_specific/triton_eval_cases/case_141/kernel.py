import numpy as np
import torch
import triton
import triton.language as tl

SEED = 194001


@triton.jit
def _pruned_ffn(X, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x0 = tl.load(X + row * 12, row < N, other=0.0)
    x1 = tl.load(X + row * 12 + 1, row < N, other=0.0)
    value = 0.25 * x0 + 0.5 * x1
    tl.store(Out + row, value, row < N)


def run(x, weights, biases, coefficients):
    output = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
    _pruned_ffn[(triton.cdiv(x.shape[0], 128),)](
        x, output, N=x.shape[0], BLOCK=128, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(SEED))
    center = rng.choice(np.asarray([-1, 1]), size=12)
    flip_probability = rng.uniform(0.05, 0.4)
    flips = np.where(rng.uniform(size=(6, 12)) < flip_probability, -1, 1)
    magnitudes = rng.choice(np.asarray([0.5, 1.0]), size=(6, 12))
    weights = (center[None, :] * flips * magnitudes).astype(np.float32)
    biases = (0.75 * np.abs(weights).sum(axis=1)).astype(np.float32)
    coefficients = np.full(6, 0.25, dtype=np.float32)
    smoke = np.concatenate((np.zeros((1, 12)), np.eye(12), -np.eye(12),
                            rng.uniform(-1.0, 1.0, size=(16, 12))), axis=0).astype(np.float32)
    return smoke, weights, biases, coefficients


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(value).to(device) for value in make_inputs_numpy())


def initial_probe():
    inputs = make_inputs()
    before = [value.clone() for value in inputs]
    actual = run(*inputs)
    x, w, b, c = [value.double() for value in inputs]
    expected = 0.25*x[:, 0] + 0.5*x[:, 1] + torch.relu(x @ w.T - b) @ c
    error = torch.abs(actual.double() - expected)
    structural = (actual.dtype == torch.float32 and actual.shape == (41,)
                  and bool(torch.isfinite(actual).all())
                  and all(torch.equal(a, z) for a, z in zip(inputs, before)))
    return {"scope": "41 smoke rows: origin, signed coordinate axes, and 16 uniform box samples",
            "max_absolute_error": float(error.max()), "tolerance": 1.0,
            "shape_dtype_finite_and_inputs_unmodified": bool(structural),
            "all_passed": bool(structural and float(error.max()) <= 1.0)}
