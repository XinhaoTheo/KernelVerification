"""Private construction and independent oracles for raw-moment LayerNorm.

Only emitted eval_cases files are given to agents. The search, labels, and
reference implementations remain outside the evaluation image.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CASES = ("case_44", "case_45")
TOLERANCE = 0.02
N = 128
OFFSET = 64.0
SCALE = 0.125
EPSILON = 1e-5
SEARCH_SEEDS = range(782401, 782657)


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    return ((OFFSET + rng.normal(0.0, SCALE, N)).astype(np.float32),)


def reference(inputs_numpy):
    """Centered FP64 population variance on the original FP32 inputs."""
    (x,) = (np.asarray(value, dtype=np.float64) for value in inputs_numpy)
    mean = np.mean(x, dtype=np.float64)
    centered = x - mean
    variance = np.mean(centered * centered, dtype=np.float64)
    return centered / np.sqrt(variance + EPSILON)


def independent_reference(inputs_numpy):
    """Scalar, translated fsum oracle; independent of NumPy reductions."""
    (array,) = inputs_numpy
    values = [float(value) for value in array]
    anchor = values[0]
    mean_shift = math.fsum(value - anchor for value in values) / len(values)
    centered = [(value - anchor) - mean_shift for value in values]
    variance = math.fsum(value * value for value in centered) / len(values)
    denominator = math.sqrt(variance + EPSILON)
    return np.asarray([value / denominator for value in centered], dtype=np.float64)


def _moments(inputs_numpy):
    (array,) = inputs_numpy
    total = np.float32(0.0)
    squares = np.float32(0.0)
    for value in np.asarray(array, dtype=np.float32):
        total = np.float32(total + value)
        squares = np.float32(squares + np.float32(value * value))
    mean = np.float32(total / np.float32(len(array)))
    second = np.float32(squares / np.float32(len(array)))
    variance = np.float32(second - np.float32(mean * mean))
    return mean, variance


def emulate(inputs_numpy):
    """Explicit FP32 accumulation with no fused multiply/add contraction."""
    (array,) = inputs_numpy
    mean, variance = _moments(inputs_numpy)
    denominator = np.sqrt(np.float32(max(variance, np.float32(0.0)) + np.float32(EPSILON)))
    return ((np.asarray(array, dtype=np.float32) - mean) / denominator).astype(np.float32)


def metric(output, reference_output):
    output = np.asarray(output, dtype=np.float64)
    exact = np.asarray(reference_output, dtype=np.float64)
    if output.shape != exact.shape or not np.all(np.isfinite(output)):
        return float("inf")
    numerator = math.sqrt(math.fsum(float(value) ** 2 for value in (output - exact).flat))
    denominator = max(math.sqrt(math.fsum(float(value) ** 2 for value in exact.flat)),
                      0.001 * math.sqrt(exact.size))
    return numerator / denominator


def _search():
    records = []
    for seed in SEARCH_SEEDS:
        inputs = _inputs(seed)
        exact = reference(inputs)
        independent = independent_reference(inputs)
        disagreement = float(np.max(np.abs(exact - independent)))
        if disagreement > 1e-12:
            raise ValueError(f"Independent oracle disagreement for seed {seed}")
        mean, raw_variance = _moments(inputs)
        (x,) = inputs
        centered = x.astype(np.float64) - np.mean(x, dtype=np.float64)
        exact_variance = float(np.mean(centered * centered))
        records.append({
            "seed": seed, "error": metric(emulate(inputs), exact),
            "fp32_mean": float(mean), "raw_variance": float(raw_variance),
            "reference_variance": exact_variance,
            "oracle_max_abs_disagreement": disagreement,
        })
    passing = min(records, key=lambda row: row["error"])
    failing = max(records, key=lambda row: row["error"])
    if passing["error"] > 0.75 * TOLERANCE or failing["error"] < 1.25 * TOLERANCE:
        raise ValueError("Search did not produce the required numerical margins")
    return records, {"case_44": passing, "case_45": failing}


PROBLEM = """Implement LayerNorm without an affine transform for this fixed workload.

The only input in scope is the float32 vector x returned by make_inputs() in
kernel.py. Its shape is (128,), and its complete contents and element order are
specified by that public deterministic generator.

The reference uses the original float32 input values, evaluated in float64:
  mean = sum_i x[i] / 128
  variance = sum_i (x[i] - mean)**2 / 128
  reference[i] = (x[i] - mean) / sqrt(variance + 0.00001)
Use population variance, with no degrees-of-freedom correction.

The output must be one finite vector with shape (128,). Its relative L2 error is
  ||output - reference||_2 / max(||reference||_2, 0.001 * sqrt(128)).
The numerical contract requires this error to be <= 0.02.

Decide whether this implementation satisfies that contract for this fixed
workload. Arbitrary alternative tensors are outside this question's scope.
"""


KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(X, Output, N: tl.constexpr):
    total = tl.full((), 0.0, tl.float32)
    squares = tl.full((), 0.0, tl.float32)
    for i in range(0, N):
        value = tl.load(X + i).to(tl.float32)
        total = total + value
        squares = squares + value * value
    mean = total / N
    variance = tl.maximum(squares / N - mean * mean, 0.0)
    denominator = tl.sqrt(variance + 0.00001)
    offsets = tl.arange(0, N)
    values = tl.load(X + offsets).to(tl.float32)
    tl.store(Output + offsets, (values - mean) / denominator)


def run(x):
    output = torch.empty((128,), device=x.device, dtype=torch.float32)
    _layer_norm_kernel[(1,)](x, output, N=128, num_warps=4,
                            enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    x = (64.0 + rng.normal(0.0, 0.125, 128)).astype(np.float32)
    return (x,)


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
'''


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write_frozen(path, text):
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"Refusing to replace an existing case/artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build(output_root: Path):
    root = Path(output_root)
    search, selected = _search()
    answer_key = {"family": "variance", "budget": TOLERANCE, "cases": {}}
    for case, selection in selected.items():
        inputs = _inputs(selection["seed"])
        exact = reference(inputs)
        approximate = emulate(inputs)
        error = metric(approximate, exact)
        code = KERNEL_TEMPLATE.replace("__SEED__", str(selection["seed"]))
        folder = root / "eval_cases" / case
        _write_frozen(folder / "kernel.py", code)
        _write_frozen(folder / "problem.txt", PROBLEM)
        _write_frozen(folder / "meta.json", json.dumps({"name": case, "passed": None,
                                                       "status": "unverified"}, indent=2) + "\n")
        answer_key["cases"][case] = {
            "cpu_ground_truth": "trust" if error <= TOLERANCE else "reject",
            "error": error, "budget": TOLERANCE,
            "kernel_sha256": _sha(code.encode()), "problem_sha256": _sha(PROBLEM.encode()),
            "input_sha256": [_sha(array.tobytes()) for array in inputs],
            "reference_norm": float(np.linalg.norm(exact)),
            "reference": exact.tolist(), "emulated_output": approximate.tolist(),
            "oracle_max_abs_disagreement": selection["oracle_max_abs_disagreement"],
            "relative_margin_to_budget": abs(error - TOLERANCE) / TOLERANCE,
            "seed": selection["seed"],
            "raw_variance": selection["raw_variance"],
            "reference_variance": selection["reference_variance"],
            "mechanism": "FP32 raw second-moment subtraction loses variance accuracy; the concrete seed determines final LayerNorm compliance.",
        }
    _write_frozen(root / "private_data" / "answer_key_variance.json", json.dumps(answer_key, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_variance.json", json.dumps({
        "family": "variance", "numpy_version": np.__version__, "budget": TOLERANCE,
        "fixed_parameters": {"N": N, "offset": OFFSET, "scale": SCALE, "epsilon": EPSILON},
        "selection_rule": "Search all 256 seeds 782401 through 782656; select minimum and maximum error, requiring at least 25% margins from the predetermined 0.02 tolerance.",
        "candidates": search,
        "selected": {case: {"seed": row["seed"]} for case, row in selected.items()},
    }, indent=2) + "\n")
    return answer_key


if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[1])
    print(json.dumps({case: {key: row[key] for key in ("cpu_ground_truth", "error", "budget",
                                                     "raw_variance", "reference_variance",
                                                     "relative_margin_to_budget", "kernel_sha256")}
                      for case, row in result["cases"].items()}, indent=2))
