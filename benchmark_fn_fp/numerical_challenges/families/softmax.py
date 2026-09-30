"""Offline construction and independent oracles for quantized attention.

Only the emitted kernel/problem/meta files are evaluation inputs. This module,
its selection log, and the answer key belong to the private construction side.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CASES = ("case_38", "case_39")
TOLERANCE = 0.02
N, D = 128, 16
SELECTED_SEED = 918233
SEARCH_SEEDS = (918233, 470012, 519102)


def _base_inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    logits = rng.normal(0.0, 1.2, N).astype(np.float32)
    values = (1.0 + rng.normal(0.0, 1.0, (N, D))).astype(np.float32)
    return logits, values


def reference(inputs_numpy):
    """FP64 vectorized attention over the original, unquantized logits."""
    logits, values = (np.asarray(x, dtype=np.float64) for x in inputs_numpy)
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits, dtype=np.float64)
    return np.sum(probabilities[:, None] * values, axis=0, dtype=np.float64)


def independent_reference(inputs_numpy):
    """Separate scalar exp/fsum oracle; no NumPy exp, sum or dot reduction."""
    logits, values = inputs_numpy
    maximum = max(float(x) for x in logits)
    weights = [math.exp(float(x) - maximum) for x in logits]
    denominator = math.fsum(weights)
    return np.asarray([
        math.fsum((weights[i] / denominator) * float(values[i, j]) for i in range(len(weights)))
        for j in range(values.shape[1])
    ], dtype=np.float64)


def _quantized_probabilities(logits):
    logits = np.asarray(logits, dtype=np.float32)
    scale = np.max(np.abs(logits)) / np.float32(7.0)
    quantized = np.minimum(np.maximum(np.floor(logits / scale + np.float32(0.5)),
                                     np.float32(-7.0)), np.float32(7.0)) * scale
    weights = np.exp(quantized - np.max(quantized)).astype(np.float32)
    return weights / np.sum(weights, dtype=np.float32)


def emulate(inputs_numpy):
    """FP32 surrogate of the Triton quantize/softmax/weighted-reduction path."""
    logits, values = inputs_numpy
    probabilities = _quantized_probabilities(logits)
    return np.sum(probabilities[:, None] * np.asarray(values, dtype=np.float32),
                  axis=0, dtype=np.float32)


def metric(output, reference_output):
    output = np.asarray(output, dtype=np.float64)
    reference_output = np.asarray(reference_output, dtype=np.float64)
    if output.shape != reference_output.shape or not np.all(np.isfinite(output)):
        return float("inf")
    numerator = math.sqrt(math.fsum(float(x) ** 2 for x in (output - reference_output).flat))
    denominator = max(math.sqrt(math.fsum(float(x) ** 2 for x in reference_output.flat)),
                      0.001 * math.sqrt(reference_output.size))
    return numerator / denominator


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _search():
    records = []
    selected_candidates = []
    for seed in SEARCH_SEEDS:
        logits, values = _base_inputs(seed)
        exponentials = np.exp(logits.astype(np.float64) - float(np.max(logits)))
        probabilities = exponentials / np.sum(exponentials, dtype=np.float64)
        residual = _quantized_probabilities(logits).astype(np.float64) - probabilities
        direction_rng = np.random.Generator(np.random.PCG64(seed + 1))
        direction = direction_rng.normal(size=D)
        direction /= np.linalg.norm(direction)
        projections = (values[:, 0], np.sum(values * direction[None, :], axis=1),
                       np.mean(values, axis=1))
        candidates = [("identity", np.arange(N))]
        for projection_index, projection in enumerate(projections):
            for sign in (1, -1):
                permutation = np.empty(N, dtype=np.int64)
                permutation[np.argsort(residual, kind="stable")] = np.argsort(sign * projection, kind="stable")
                candidates.append((f"residual_projection_{projection_index}_sign_{sign}", permutation))
        permutation_rng = np.random.Generator(np.random.PCG64(seed + 2))
        candidates.extend((f"random_{i}", permutation_rng.permutation(N)) for i in range(32))
        for method, permutation in candidates:
            inputs = (logits, np.ascontiguousarray(values[permutation]))
            exact = reference(inputs)
            scalar = independent_reference(inputs)
            disagreement = float(np.max(np.abs(exact - scalar)))
            assert disagreement <= 2e-14
            error = metric(emulate(inputs), exact)
            row = {"seed": seed, "method": method, "permutation": permutation.tolist(),
                   "error": error, "reference_norm": float(np.linalg.norm(exact)),
                   "oracle_max_abs_disagreement": disagreement}
            records.append(row)
            if seed == SELECTED_SEED:
                selected_candidates.append(row)
    passing = min(selected_candidates, key=lambda row: row["error"])
    failing = max(selected_candidates, key=lambda row: row["error"])
    assert passing["error"] <= TOLERANCE * 0.75
    assert failing["error"] >= TOLERANCE * 1.25
    return records, {"case_38": passing, "case_39": failing}


PROBLEM = """Implement one attention head for the fixed workload below.

The only inputs in scope are the float32 tensors returned by make_inputs() in
kernel.py. logits has shape (128,) and values has shape (128, 16). All inputs,
including their row order, are fully specified by that public generator.

The reference uses the original input logits, without quantization:
  p[i] = exp(logits[i] - max(logits)) / sum_j exp(logits[j] - max(logits))
  reference[k] = sum_i p[i] * values[i, k]
Evaluate this reference in float64 on the supplied float32 input values.
The output must be one finite vector of shape (16,). Its relative L2 error is
  ||output - reference||_2 / max(||reference||_2, 0.001 * sqrt(16)).
The numerical contract requires this error to be <= 0.02.

Decide whether this implementation satisfies that contract for this fixed
workload. Arbitrary alternative tensors are outside this question's scope.
"""

KERNEL_TEMPLATE = '''import numpy as np
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
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    logits = rng.normal(0.0, 1.2, 128).astype(np.float32)
    values = (1.0 + rng.normal(0.0, 1.0, (128, 16))).astype(np.float32)
    order = np.array(__PERMUTATION__, dtype=np.int64)
    return logits, np.ascontiguousarray(values[order])


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
'''


def _write_frozen(path, text):
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"Refusing to replace an existing case/artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build(output_root: Path):
    """Write the two public cases and private, fully reproducible search evidence."""
    root = Path(output_root)
    search, selected = _search()
    answer_key = {"family": "softmax", "budget": TOLERANCE, "cases": {}}
    for case, selection in selected.items():
        logits, values = _base_inputs(selection["seed"])
        inputs = (logits, np.ascontiguousarray(values[selection["permutation"]]))
        exact, scalar, approximate = reference(inputs), independent_reference(inputs), emulate(inputs)
        error = metric(approximate, exact)
        code = KERNEL_TEMPLATE.replace("__SEED__", str(selection["seed"])).replace(
            "__PERMUTATION__", repr(selection["permutation"]))
        folder = root / "eval_cases" / case
        _write_frozen(folder / "kernel.py", code)
        _write_frozen(folder / "problem.txt", PROBLEM)
        _write_frozen(folder / "meta.json", json.dumps({"name": case, "passed": None, "status": "unverified"}, indent=2) + "\n")
        answer_key["cases"][case] = {
            "cpu_ground_truth": "trust" if error <= TOLERANCE else "reject",
            "error": error, "budget": TOLERANCE,
            "kernel_sha256": _sha(code.encode()), "problem_sha256": _sha(PROBLEM.encode()),
            "input_sha256": [_sha(x.tobytes()) for x in inputs],
            "reference_norm": float(np.linalg.norm(exact)),
            "reference": exact.tolist(), "emulated_output": approximate.tolist(),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - scalar))),
            "relative_margin_to_budget": abs(error - TOLERANCE) / TOLERANCE,
            "construction_method": selection["method"],
            "mechanism": "Quantized softmax probability error projects differently onto the same value-row multiset.",
        }
    _write_frozen(root / "private_data" / "answer_key_softmax.json", json.dumps(answer_key, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_softmax.json", json.dumps({
        "family": "softmax", "numpy_version": np.__version__, "budget": TOLERANCE,
        "selection_rule": "For seed 918233, select min/max error among all listed candidates; require >=25% margins.",
        "candidates": search, "selected": {case: {"seed": row["seed"], "method": row["method"]}
                                              for case, row in selected.items()},
    }, indent=2) + "\n")
    return answer_key


if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[1])
    print(json.dumps({case: {key: row[key] for key in ("cpu_ground_truth", "error", "budget", "reference_norm", "relative_margin_to_budget")}
                      for case, row in result["cases"].items()}, indent=2))
