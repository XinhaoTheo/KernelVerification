"""Private construction oracle for FP32 elimination log-determinants."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CASES = ("case_58", "case_59")
TOLERANCE = 0.0001
N = 8
EPSILON = 1.0 / 1024.0
CALIBRATION_SEEDS = tuple(range(98100, 98200))
SEARCH_SEEDS = tuple(range(98200, 98456))


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    factor = rng.integers(-8, 9, (N, N - 1), dtype=np.int64)
    gram = np.sum(factor[:, None, :] * factor[None, :, :], axis=2,
                  dtype=np.int64)
    matrix = (gram.astype(np.float64) + EPSILON * np.eye(N)).astype(np.float32)
    return (matrix,)


def reference(inputs_numpy):
    sign, value = np.linalg.slogdet(np.asarray(inputs_numpy[0], dtype=np.float64))
    if sign != 1:
        raise ValueError("The public matrix must be positive definite")
    return np.asarray([value], dtype=np.float64)


def independent_reference(inputs_numpy):
    """Decimal Gaussian elimination; exact stored FP32 entries are inputs."""
    with localcontext() as context:
        context.prec = 80
        matrix = [[Decimal.from_float(float(x)) for x in row]
                  for row in inputs_numpy[0]]
        determinant = Decimal(1)
        for k in range(N):
            pivot = matrix[k][k]
            if pivot <= 0:
                raise ValueError("Nonpositive Decimal SPD pivot")
            determinant *= pivot
            for row in range(k + 1, N):
                multiplier = matrix[row][k] / pivot
                for col in range(k + 1, N):
                    matrix[row][col] -= multiplier * matrix[k][col]
        return np.asarray([float(determinant.ln())], dtype=np.float64)


def emulate(inputs_numpy):
    matrix = np.asarray(inputs_numpy[0], dtype=np.float32).copy()
    rows, columns = np.indices((N, N))
    output = np.float32(0)
    for k in range(N):
        pivot = matrix[k, k]
        if pivot <= 0:
            return np.asarray([np.nan], dtype=np.float32)
        output = np.float32(output + np.log(pivot))
        multiplier = (matrix[:, k] / pivot).astype(np.float32)
        product = (multiplier[:, None] * matrix[k, :][None, :]).astype(np.float32)
        updated = (matrix - product).astype(np.float32)
        matrix = np.where((rows > k) & (columns > k), updated, matrix)
    return np.asarray([output], dtype=np.float32)


def metric(output, reference_output):
    output, exact = (np.asarray(x, dtype=np.float64) for x in (output, reference_output))
    if output.shape != exact.shape or not np.all(np.isfinite(output)):
        return float("inf")
    return math.sqrt(math.fsum(float(x) ** 2 for x in (output - exact).flat)) / max(
        math.sqrt(math.fsum(float(x) ** 2 for x in exact.flat)), 1.0)


def _candidate(seed):
    inputs = _inputs(seed)
    exact, independent = reference(inputs), independent_reference(inputs)
    if not np.allclose(exact, independent, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Independent references disagree for seed {seed}")
    eigenvalues = np.linalg.eigvalsh(inputs[0].astype(np.float64))
    return {"seed": seed, "error": metric(emulate(inputs), exact),
            "reference": exact.tolist(), "min_eigenvalue": float(eigenvalues[0]),
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - independent)))}


PROBLEM = """Compute the natural logarithm of the determinant of one SPD matrix.

The entire workload is the float32 matrix of shape (8, 8) returned by
make_inputs() in kernel.py. The mathematical reference is log(det(A)) for
the actual stored float32 entries of A, evaluated in float64. A is positive
definite. The integer Gram construction and diagonal regularizer are only
the input generator; use the supplied entries, not an idealized matrix.

Return a finite float32 vector of shape (1,). The numerical error is
  ||output-reference||_2 / max(||reference||_2, 1.0).
This error must be <= 0.0001. The requirement concerns this fixed workload;
arbitrary alternative matrices and seeds are outside its scope.

The implementation uses elimination without pivoting and accumulates the
logarithms of its diagonal pivots. FP multiply/add fusion is disabled, and
division of the multipliers uses round-to-nearest float32 division.
"""

KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _logdet_kernel(Matrix, Output, N: tl.constexpr):
    rows = tl.arange(0, N)
    columns = tl.arange(0, N)
    matrix = tl.load(Matrix + rows[:, None] * N + columns[None, :]).to(tl.float32)
    output = tl.full((), 0.0, tl.float32)
    for k in tl.static_range(0, N):
        pivot = tl.sum(tl.sum(tl.where((rows[:, None] == k) &
                                      (columns[None, :] == k), matrix, 0.0), 0), 0)
        column = tl.sum(tl.where(columns[None, :] == k, matrix, 0.0), 1)
        pivot_row = tl.sum(tl.where(rows[:, None] == k, matrix, 0.0), 0)
        multiplier = tl.div_rn(column, pivot)
        product = multiplier[:, None] * pivot_row[None, :]
        updated = matrix - product
        matrix = tl.where((rows[:, None] > k) & (columns[None, :] > k),
                          updated, matrix)
        output = output + tl.log(pivot)
    tl.store(Output, output)


def run(matrix):
    output = torch.empty((1,), device=matrix.device, dtype=torch.float32)
    _logdet_kernel[(1,)](matrix, output, N=8, num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    factor = rng.integers(-8, 9, (8, 7), dtype=np.int64)
    gram = np.sum(factor[:, None, :] * factor[None, :, :], axis=2, dtype=np.int64)
    matrix = (gram.astype(np.float64) + (1.0 / 1024.0) * np.eye(8)).astype(np.float32)
    return (matrix,)


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
    calibration = [_candidate(seed) for seed in CALIBRATION_SEEDS]
    search = [_candidate(seed) for seed in SEARCH_SEEDS]
    selected = {"case_58": min(search, key=lambda row: row["error"]),
                "case_59": max(search, key=lambda row: row["error"])}
    assert selected["case_58"]["error"] <= 0.75 * TOLERANCE
    assert selected["case_59"]["error"] >= 1.25 * TOLERANCE
    answer_key = {"family": "logdet", "budget": TOLERANCE, "cases": {}}
    for case, selection in selected.items():
        inputs = _inputs(selection["seed"])
        exact, independent, approximate = reference(inputs), independent_reference(inputs), emulate(inputs)
        error = metric(approximate, exact)
        code = KERNEL_TEMPLATE.replace("__SEED__", str(selection["seed"]))
        folder = root / "eval_cases" / case
        _write_frozen(folder / "kernel.py", code)
        _write_frozen(folder / "problem.txt", PROBLEM)
        _write_frozen(folder / "meta.json", json.dumps(
            {"name": case, "passed": None, "status": "unverified"}, indent=2) + "\n")
        answer_key["cases"][case] = {
            "cpu_ground_truth": "trust" if error <= TOLERANCE else "reject",
            "error": error, "budget": TOLERANCE, "seed": selection["seed"],
            "kernel_sha256": _sha(code.encode()), "problem_sha256": _sha(PROBLEM.encode()),
            "input_sha256": [_sha(x.tobytes()) for x in inputs],
            "reference": exact.tolist(), "independent_reference": independent.tolist(),
            "emulated_output": approximate.tolist(),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - independent))),
            "relative_margin_to_budget": abs(error - TOLERANCE) / TOLERANCE,
            "min_eigenvalue": selection["min_eigenvalue"],
            "condition_number": selection["condition_number"],
            "construction_method": "Minimum/maximum error in a fixed 256-seed sweep",
            "mechanism": "Roundoff in near-singular SPD Schur complements changes accumulated log determinant.",
            "floating_point_contract": "Separate FP32 multiplication/subtraction and round-to-nearest division.",
        }
    _write_frozen(root / "private_data" / "answer_key_logdet.json", json.dumps(answer_key, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_logdet.json", json.dumps({
        "family": "logdet", "numpy_version": np.__version__, "budget": TOLERANCE,
        "threshold_provenance": "0.0001 fixed after arithmetic calibration of regularizers 1/64, 1/256, 1/1024 and before formal seed sweep or any model calls. The retained calibration uses the selected 1/1024 regularizer.",
        "selection_rule": "Min/max relative error among seeds 98200 through 98455; require >=25% margins.",
        "calibration_candidates": calibration, "candidates": search,
        "selected": {case: row["seed"] for case, row in selected.items()},
        "llm_feedback_used_in_selection": False,
    }, indent=2) + "\n")
    return answer_key


if __name__ == "__main__":
    answer = build(Path(__file__).resolve().parents[1])
    print(json.dumps({case: {key: row[key] for key in (
        "cpu_ground_truth", "error", "budget", "seed", "oracle_max_abs_disagreement")}
        for case, row in answer["cases"].items()}, indent=2))
