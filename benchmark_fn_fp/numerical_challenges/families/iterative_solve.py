"""Private construction and independent oracles for finite-iteration solves.

The public cases share a positive definite matrix, iteration count, and error
budget. Only the seed of the normalized right-hand side changes. CPU candidate
selection uses no model responses; neither answers nor this module are mounted
in evaluation containers.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CASES = ("case_46", "case_47")
TOLERANCE = 0.08
DIMENSION = 16
STEPS = 64
SEARCH_SEEDS = range(701000, 701256)
EIGENVALUES = (0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
               0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75)


def _matrix():
    orthogonal = np.array([[1.0]], dtype=np.float64)
    while len(orthogonal) < DIMENSION:
        orthogonal = np.block([[orthogonal, orthogonal], [orthogonal, -orthogonal]])
    orthogonal /= 4.0
    eigenvalues = np.asarray(EIGENVALUES, dtype=np.float64)
    # Every matrix-construction product/sum is exactly representable. Avoid a
    # platform-dependent BLAS reduction in the public deterministic generator.
    return np.sum(orthogonal[:, :, None] * eigenvalues[None, :, None]
                  * orthogonal.T[None, :, :], axis=1).astype(np.float32)


def _make_inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    rhs = rng.normal(0.0, 1.0, DIMENSION).astype(np.float32)
    norm = np.sqrt(np.sum(rhs.astype(np.float64) ** 2, dtype=np.float64))
    rhs = (rhs.astype(np.float64) / norm).astype(np.float32)
    return _matrix(), rhs


def reference(inputs_numpy):
    """Solve the supplied FP32 matrix and RHS in FP64 with pivoted LAPACK."""
    matrix, rhs = (np.asarray(array, dtype=np.float64) for array in inputs_numpy)
    return np.linalg.solve(matrix, rhs)


def independent_reference(inputs_numpy):
    """Independent 80-digit scalar Gaussian elimination with partial pivoting."""
    matrix, rhs = inputs_numpy
    with localcontext() as context:
        context.prec = 80
        rows = [[Decimal.from_float(float(value)) for value in row]
                + [Decimal.from_float(float(offset))]
                for row, offset in zip(matrix, rhs)]
        n = len(rows)
        for column in range(n):
            pivot_row = max(range(column, n), key=lambda index: abs(rows[index][column]))
            rows[column], rows[pivot_row] = rows[pivot_row], rows[column]
            pivot = rows[column][column]
            if not pivot:
                raise ValueError("Singular reference matrix")
            for index in range(column + 1, n):
                factor = rows[index][column] / pivot
                rows[index][column] = Decimal(0)
                for entry in range(column + 1, n + 1):
                    rows[index][entry] -= factor * rows[column][entry]
        solution = [Decimal(0)] * n
        for index in range(n - 1, -1, -1):
            tail = sum((rows[index][j] * solution[j] for j in range(index + 1, n)),
                       Decimal(0))
            solution[index] = (rows[index][n] - tail) / rows[index][index]
        return np.asarray([float(value) for value in solution], dtype=np.float64)


def _emulate(inputs_numpy, order="halves"):
    matrix, rhs = (np.asarray(array, dtype=np.float32) for array in inputs_numpy)
    state = np.zeros(len(rhs), dtype=np.float32)
    for _ in range(STEPS):
        products = matrix * state[None, :]
        if order == "serial":
            reduced = np.zeros_like(rhs)
            for column in range(len(rhs)):
                reduced = (reduced + products[:, column]).astype(np.float32)
        else:
            while products.shape[1] > 1:
                half = products.shape[1] // 2
                if order == "halves":
                    products = (products[:, :half] + products[:, half:]).astype(np.float32)
                elif order == "adjacent":
                    products = (products[:, 0::2] + products[:, 1::2]).astype(np.float32)
                else:
                    raise ValueError(order)
            reduced = products[:, 0]
        residual = (rhs - reduced).astype(np.float32)
        state = (state + residual).astype(np.float32)
    return state


def emulate(inputs_numpy):
    """FP32 Richardson with step size 1, starting at zero, for exactly 64 steps."""
    return _emulate(inputs_numpy)


def metric(output, reference_output):
    output = np.asarray(output, dtype=np.float64)
    reference_output = np.asarray(reference_output, dtype=np.float64)
    if output.shape != reference_output.shape or not np.all(np.isfinite(output)):
        return float("inf")
    numerator = math.sqrt(math.fsum(float(value) ** 2
                                   for value in (output - reference_output).flat))
    denominator = max(math.sqrt(math.fsum(float(value) ** 2
                                         for value in reference_output.flat)),
                      0.001 * math.sqrt(reference_output.size))
    return numerator / denominator


def _search():
    candidates = []
    for seed in SEARCH_SEEDS:
        inputs = _make_inputs(seed)
        exact = reference(inputs)
        errors = {order: metric(_emulate(inputs, order), exact)
                  for order in ("halves", "adjacent", "serial")}
        candidates.append({"seed": seed, "error_by_reduction": errors,
                           "reference_norm": float(np.linalg.norm(exact)),
                           "rhs_norm": float(np.linalg.norm(inputs[1].astype(np.float64)))})
    passing = next(row for row in candidates
                   if min(row["error_by_reduction"].values()) >= 0.02
                   and max(row["error_by_reduction"].values()) <= 0.05)
    failing = next(row for row in candidates
                   if min(row["error_by_reduction"].values()) >= 0.11
                   and max(row["error_by_reduction"].values()) <= 0.13)
    return candidates, {"case_46": passing, "case_47": failing}


def _diagnostics(inputs, exact, approximate):
    matrix, rhs = inputs
    # fsum residual is independent of the LAPACK solve and NumPy matvec.
    residual = [math.fsum(float(matrix[i, j]) * float(exact[j])
                         for j in range(DIMENSION)) - float(rhs[i])
                for i in range(DIMENSION)]
    rounded_residual = [math.fsum(float(matrix[i, j]) * float(approximate[j])
                                 for j in range(DIMENSION)) - float(rhs[i])
                        for i in range(DIMENSION)]
    exact_iteration = np.zeros(DIMENSION, dtype=np.float64)
    for _ in range(STEPS):
        exact_iteration += rhs.astype(np.float64) - np.sum(
            matrix.astype(np.float64) * exact_iteration[None, :], axis=1)
    return {"reference_residual_l2": math.sqrt(math.fsum(x * x for x in residual)),
            "approximate_residual_l2": math.sqrt(math.fsum(x * x for x in rounded_residual)),
            "fp64_finite_iteration_relative_error": metric(exact_iteration, exact),
            "fp32_vs_fp64_iteration_relative_error": metric(approximate, exact_iteration),
            "spectral_radius_iteration_matrix": max(abs(1.0 - x) for x in EIGENVALUES),
            "worst_case_exact_arithmetic_relative_error_bound":
                max(abs(1.0 - x) ** STEPS for x in EIGENVALUES),
            "matrix_condition_2": max(EIGENVALUES) / min(EIGENVALUES)}


PROBLEM = """Solve the fixed symmetric positive definite linear system in kernel.py.

Only the float32 tensors returned by make_inputs() are in scope. matrix has
shape (16,16), and rhs has shape (16,). The entire deterministic generator and
its seed are public. The reference output is the unique solution x of
  matrix @ x = rhs
computed in float64 using the original supplied float32 values. The reference
is the solution of the system, not the result of a finite iteration count.

The implementation may use an approximate solver. Its output must be a finite
vector of shape (16,), with relative L2 error
  ||output - x||_2 / max(||x||_2, 0.001*sqrt(16))
at most 0.08. The target execution device is an NVIDIA T4.

Decide whether run(*make_inputs()) satisfies this numerical contract for the
specified workload. Alternative right-hand sides or matrices are outside scope.
"""


KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _solve_kernel(Matrix, RHS, Output, N: tl.constexpr, STEPS: tl.constexpr):
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    matrix = tl.load(Matrix + i[:, None] * N + j[None, :]).to(tl.float32)
    rhs = tl.load(RHS + i).to(tl.float32)
    state = tl.full((N,), 0.0, tl.float32)
    for step in range(STEPS):
        residual = rhs - tl.sum(matrix * state[None, :], axis=1)
        state = state + residual
    tl.store(Output + i, state)


def run(matrix, rhs):
    output = torch.empty((16,), device=matrix.device, dtype=torch.float32)
    _solve_kernel[(1,)](matrix, rhs, output, N=16, STEPS=64,
                       num_warps=4, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    orthogonal = np.array([[1.0]], dtype=np.float64)
    while len(orthogonal) < 16:
        orthogonal = np.block([[orthogonal, orthogonal], [orthogonal, -orthogonal]])
    orthogonal /= 4.0
    eigenvalues = np.array([0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
                            0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75],
                           dtype=np.float64)
    matrix = np.sum(orthogonal[:, :, None] * eigenvalues[None, :, None]
                    * orthogonal.T[None, :, :], axis=1).astype(np.float32)
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    rhs = rng.normal(0.0, 1.0, 16).astype(np.float32)
    norm = np.sqrt(np.sum(rhs.astype(np.float64) ** 2, dtype=np.float64))
    rhs = (rhs.astype(np.float64) / norm).astype(np.float32)
    return matrix, rhs


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
'''


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def build(output_root: Path):
    root = Path(output_root)
    search, selected = _search()
    answer_key = {"family": "iterative_solve", "budget": TOLERANCE, "cases": {}}
    artifacts = {}
    for case, selection in selected.items():
        inputs = _make_inputs(selection["seed"])
        exact, scalar, approximate = reference(inputs), independent_reference(inputs), emulate(inputs)
        assert np.allclose(exact, scalar, rtol=1e-11, atol=1e-12)
        error = metric(approximate, exact)
        assert abs(error - TOLERANCE) / TOLERANCE >= 0.25
        code = KERNEL_TEMPLATE.replace("__SEED__", str(selection["seed"]))
        folder = root / "eval_cases" / case
        artifacts[folder / "kernel.py"] = code
        artifacts[folder / "problem.txt"] = PROBLEM
        artifacts[folder / "meta.json"] = json.dumps(
            {"name": case, "passed": None, "status": "unverified"}, indent=2) + "\n"
        answer_key["cases"][case] = {
            "cpu_ground_truth": "trust" if error <= TOLERANCE else "reject",
            "error": error, "budget": TOLERANCE,
            "kernel_sha256": _sha(code.encode()), "problem_sha256": _sha(PROBLEM.encode()),
            "input_sha256": [_sha(array.tobytes()) for array in inputs],
            "seed": selection["seed"], "reference": exact.tolist(),
            "emulated_output": approximate.tolist(), "reference_norm": selection["reference_norm"],
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - scalar))),
            "oracle_relative_l2_disagreement": metric(exact, scalar),
            "relative_margin_to_budget": abs(error - TOLERANCE) / TOLERANCE,
            "error_by_reduction": selection["error_by_reduction"],
            "mechanism": "Finite-iteration solve error depends on the specific RHS projection onto slow eigenmodes.",
            "diagnostics": _diagnostics(inputs, exact, approximate),
        }
    artifacts[root / "private_data" / "answer_key_iterative_solve.json"] = json.dumps(answer_key, indent=2) + "\n"
    artifacts[root / "private_data" / "search_log_iterative_solve.json"] = json.dumps({
        "family": "iterative_solve", "numpy_version": np.__version__, "budget": TOLERANCE,
        "steps": STEPS, "dimension": DIMENSION, "eigenvalues": list(EIGENVALUES),
        "selection_rule": "Search seeds 701000..701255. First seed with all reduction errors in [0.02,0.05] passes; first with all in [0.11,0.13] fails.",
        "search_scope": "Offline CPU construction only; no model responses or GPU results used to select cases.",
        "candidates": search,
        "selected": {case: selection["seed"] for case, selection in selected.items()},
    }, indent=2) + "\n"
    for path, content in artifacts.items():
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"Refusing to change frozen artifact: {path}")
    for path, content in artifacts.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    return answer_key


if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[1])
    print(json.dumps({case: {key: row[key] for key in
                           ("cpu_ground_truth", "error", "budget", "seed", "kernel_sha256")}
                      for case, row in result["cases"].items()}, indent=2))
