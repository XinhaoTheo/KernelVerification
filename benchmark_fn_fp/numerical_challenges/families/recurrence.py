"""Offline construction for a finite-workload recurrent state accuracy pair.

The public kernels differ only in their input seed. They apply the same stable
but non-normal transition 64 times, rounding the state to FP16 after each step.
The answer key and the seed search stay outside the evaluation input folders.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CASES = ("case_40", "case_41")
TOLERANCE = 0.002
DIMENSION = 16
STEPS = 64
SEARCH_SEEDS = range(202000, 202512)
MATRIX_SEED = 20260923


def _transition():
    # All factors are dyadic, including the normalized Hadamard matrix. These
    # explicit sums avoid platform-specific BLAS choices in the input builder.
    orthogonal = np.array([[1.0]], dtype=np.float64)
    while len(orthogonal) < DIMENSION:
        orthogonal = np.block([[orthogonal, orthogonal], [orthogonal, -orthogonal]])
    orthogonal /= 4.0
    rng = np.random.Generator(np.random.PCG64(MATRIX_SEED))
    diagonal = rng.choice(np.array([0.875, 0.90625, 0.9375, 0.96875]), size=DIMENSION)
    upper = np.diag(diagonal)
    upper += np.diag(np.full(DIMENSION - 1, 0.1875), 1)
    upper += np.diag(rng.choice([-0.03125, 0.03125], size=DIMENSION - 2), 2)
    left = np.sum(orthogonal[:, :, None] * upper[None, :, :], axis=1)
    return np.sum(left[:, :, None] * orthogonal.T[None, :, :], axis=1).astype(np.float32)


def _make_inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = rng.normal(0.0, 0.1, DIMENSION).astype(np.float32)
    drive = rng.normal(0.0, 0.1, (STEPS, DIMENSION)).astype(np.float32)
    return _transition(), initial, drive


def reference(inputs_numpy):
    """FP64 recurrence using the exact supplied FP32 input values."""
    matrix, initial, drive = (np.asarray(array, dtype=np.float64) for array in inputs_numpy)
    state = initial.copy()
    for forcing in drive:
        state = np.sum(matrix * state[None, :], axis=1, dtype=np.float64) + forcing
    return state


def independent_reference(inputs_numpy):
    """Independent 80-digit scalar Decimal recurrence, with no NumPy reductions."""
    matrix, initial, drive = inputs_numpy
    with localcontext() as context:
        context.prec = 80
        exact_matrix = [[Decimal.from_float(float(value)) for value in row] for row in matrix]
        state = [Decimal.from_float(float(value)) for value in initial]
        for forcing in drive:
            state = [sum((coefficient * value for coefficient, value in zip(row, state)),
                         Decimal.from_float(float(offset)))
                     for row, offset in zip(exact_matrix, forcing)]
        return np.asarray([float(value) for value in state], dtype=np.float64)


def _emulate_states(inputs_numpy, order="halves"):
    matrix, initial, drive = (np.asarray(array, dtype=np.float32) for array in inputs_numpy)
    state = initial.copy()
    history = [state.copy()]
    for forcing in drive:
        products = matrix * state[None, :]
        if order == "serial":
            reduced = np.zeros(matrix.shape[0], dtype=np.float32)
            for column in range(matrix.shape[1]):
                reduced = (reduced + products[:, column]).astype(np.float32)
        else:
            while products.shape[1] > 1:
                half = products.shape[1] // 2
                if order == "adjacent":
                    products = (products[:, 0::2] + products[:, 1::2]).astype(np.float32)
                elif order == "halves":
                    products = (products[:, :half] + products[:, half:]).astype(np.float32)
                else:
                    raise ValueError(order)
            reduced = products[:, 0]
        state = (reduced + forcing).astype(np.float16).astype(np.float32)
        history.append(state.copy())
    return np.asarray(history)


def emulate(inputs_numpy):
    """FP32 pairwise matvec plus per-step FP16 round-to-nearest state."""
    return _emulate_states(inputs_numpy)[-1]


def metric(output, reference_output):
    output = np.asarray(output, dtype=np.float64)
    reference_output = np.asarray(reference_output, dtype=np.float64)
    if output.shape != reference_output.shape or not np.all(np.isfinite(output)):
        return float("inf")
    numerator = math.sqrt(math.fsum(float(value) ** 2 for value in (output - reference_output).flat))
    denominator = max(math.sqrt(math.fsum(float(value) ** 2 for value in reference_output.flat)),
                      0.001 * math.sqrt(reference_output.size))
    return numerator / denominator


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _search():
    records = []
    for seed in SEARCH_SEEDS:
        inputs = _make_inputs(seed)
        exact = reference(inputs)
        outputs = [_emulate_states(inputs, order)[-1] for order in ("halves", "adjacent", "serial")]
        errors = [metric(output, exact) for output in outputs]
        records.append({
            "seed": seed,
            "error_by_reduction": dict(zip(("halves", "adjacent", "serial"), errors)),
            "reference_norm": math.sqrt(math.fsum(float(value) ** 2 for value in exact)),
            "reduction_outputs_identical": all(np.array_equal(outputs[0], output) for output in outputs[1:]),
        })
    # Avoid either numerical boundary and keep the passing control nontrivial.
    passing = next(row for row in records
                   if 0.0005 <= min(row["error_by_reduction"].values())
                   and max(row["error_by_reduction"].values()) <= 0.001)
    failing = next(row for row in records
                   if 0.004 <= min(row["error_by_reduction"].values())
                   and max(row["error_by_reduction"].values()) <= 0.008)
    return records, {"case_40": passing, "case_41": failing}


def _propagation_diagnostics(inputs):
    matrix, _, drive = (np.asarray(array, dtype=np.float64) for array in inputs)
    states = _emulate_states(inputs).astype(np.float64)
    carried = np.zeros(DIMENSION, dtype=np.float64)
    residual_norms = []
    for index, forcing in enumerate(drive):
        unrounded = np.sum(matrix * states[index][None, :], axis=1) + forcing
        residual = states[index + 1] - unrounded
        residual_norms.append(float(np.linalg.norm(residual)))
        carried = np.sum(matrix * carried[None, :], axis=1) + residual
    actual_error = states[-1] - reference(inputs)
    return {"max_intermediate_abs_state": float(np.max(np.abs(states))),
            "local_rounding_l2_norms": residual_norms,
            "transported_error": carried.tolist(),
            "transport_identity_max_abs_disagreement": float(np.max(np.abs(carried - actual_error)))}


PROBLEM = """Evaluate a recurrent linear state update on the fixed workload in kernel.py.

The only inputs in scope are the float32 tensors returned by make_inputs():
matrix has shape (16,16), initial has shape (16,), and drive has shape (64,16).
The full input generator and its seeds are public. The desired output is the
final state of this recurrence, evaluated in float64 on those float32 inputs:
  reference_state = initial
  for t in range(64):
      reference_state = matrix @ reference_state + drive[t]

The implementation may approximate intermediate states. Its final output must
be a finite vector of shape (16,), with relative L2 error
  ||output - reference_state||_2 / max(||reference_state||_2, 0.001*sqrt(16))
at most 0.002. The target execution device is an NVIDIA T4.

Decide whether run(*make_inputs()) satisfies this numerical contract for the
specified workload. Alternative inputs are outside this question's scope.
"""


KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _recurrence_kernel(Matrix, Initial, Drive, Output, N: tl.constexpr, STEPS: tl.constexpr):
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    matrix = tl.load(Matrix + i[:, None] * N + j[None, :])
    state = tl.load(Initial + i).to(tl.float32)
    for step in range(STEPS):
        forcing = tl.load(Drive + step * N + i)
        next_state = tl.sum(matrix * state[None, :], axis=1) + forcing
        state = next_state.to(tl.float16).to(tl.float32)
    tl.store(Output + i, state)


def run(matrix, initial, drive):
    output = torch.empty((16,), device=matrix.device, dtype=torch.float32)
    _recurrence_kernel[(1,)](matrix, initial, drive, output, N=16, STEPS=64,
                            num_warps=4, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    orthogonal = np.array([[1.0]], dtype=np.float64)
    while len(orthogonal) < 16:
        orthogonal = np.block([[orthogonal, orthogonal], [orthogonal, -orthogonal]])
    orthogonal /= 4.0
    matrix_rng = np.random.Generator(np.random.PCG64(20260923))
    diagonal = matrix_rng.choice(np.array([0.875, 0.90625, 0.9375, 0.96875]), size=16)
    upper = np.diag(diagonal)
    upper += np.diag(np.full(15, 0.1875), 1)
    upper += np.diag(matrix_rng.choice([-0.03125, 0.03125], size=14), 2)
    left = np.sum(orthogonal[:, :, None] * upper[None, :, :], axis=1)
    matrix = np.sum(left[:, :, None] * orthogonal.T[None, :, :], axis=1).astype(np.float32)
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    initial = rng.normal(0.0, 0.1, 16).astype(np.float32)
    drive = rng.normal(0.0, 0.1, (64, 16)).astype(np.float32)
    return matrix, initial, drive


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
'''


def _write_frozen(path, content):
    if path.exists() and path.read_text() != content:
        raise FileExistsError(f"Refusing to change an existing frozen case/artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content)


def build(output_root: Path):
    root = Path(output_root)
    search, selected = _search()
    answer_key = {"family": "recurrence", "budget": TOLERANCE, "cases": {}}
    artifacts = {}
    for case, selection in selected.items():
        inputs = _make_inputs(selection["seed"])
        exact = reference(inputs)
        scalar = independent_reference(inputs)
        approximate = emulate(inputs)
        assert np.allclose(exact, scalar, rtol=1e-11, atol=1e-12)
        error = metric(approximate, exact)
        assert abs(error - TOLERANCE) / TOLERANCE >= 0.25
        code = KERNEL_TEMPLATE.replace("__SEED__", str(selection["seed"]))
        folder = root / "eval_cases" / case
        artifacts[folder / "kernel.py"] = code
        artifacts[folder / "problem.txt"] = PROBLEM
        artifacts[folder / "meta.json"] = json.dumps({"name": case, "passed": None, "status": "unverified"}, indent=2) + "\n"
        answer_key["cases"][case] = {
            "cpu_ground_truth": "trust" if error <= TOLERANCE else "reject",
            "error": error, "budget": TOLERANCE,
            "kernel_sha256": _sha(code.encode()), "problem_sha256": _sha(PROBLEM.encode()),
            "input_sha256": [_sha(array.tobytes()) for array in inputs],
            "seed": selection["seed"], "reference": exact.tolist(),
            "emulated_output": approximate.tolist(),
            "reference_norm": selection["reference_norm"],
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - scalar))),
            "oracle_relative_l2_disagreement": metric(exact, scalar),
            "relative_margin_to_budget": abs(error - TOLERANCE) / TOLERANCE,
            "error_by_reduction": selection["error_by_reduction"],
            "reduction_outputs_identical": selection["reduction_outputs_identical"],
            "mechanism": "Per-step FP16 state rounding propagated by powers of the same stable non-normal matrix.",
            "diagnostics": _propagation_diagnostics(inputs),
        }
    artifacts[root / "private_data" / "answer_key_recurrence.json"] = json.dumps(answer_key, indent=2) + "\n"
    artifacts[root / "private_data" / "search_log_recurrence.json"] = json.dumps({
        "family": "recurrence", "numpy_version": np.__version__, "budget": TOLERANCE,
        "matrix_seed": MATRIX_SEED, "steps": STEPS, "dimension": DIMENSION,
        "selection_rule": "Search seeds 202000..202511. First seed with every reduction error in [0.0005,0.001] is passing; first in [0.004,0.008] is failing.",
        "search_scope": "Offline CPU development search; no model responses or GPU measurements used in selection.",
        "candidates": search,
        "selected": {case: selection["seed"] for case, selection in selected.items()},
    }, indent=2) + "\n"
    # Preflight every destination so a rerun cannot partially replace a frozen
    # family after any paid model evaluations have started.
    for path, content in artifacts.items():
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"Refusing to change an existing frozen case/artifact: {path}")
    for path, content in artifacts.items():
        _write_frozen(path, content)
    return answer_key


if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[1])
    print(json.dumps({case: {key: row[key] for key in (
        "cpu_ground_truth", "error", "budget", "relative_margin_to_budget",
        "oracle_max_abs_disagreement", "oracle_relative_l2_disagreement")}
        for case, row in result["cases"].items()}, indent=2))
