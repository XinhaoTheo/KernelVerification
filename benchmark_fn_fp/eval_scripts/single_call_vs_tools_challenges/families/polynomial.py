"""Fixed-input polynomial evaluation with data-dependent conditioning.

Private construction/oracle module. Only emitted triton_eval_cases are model inputs.
Both references evaluate the actual FP32 input coefficients, including c[0]
after its initial FP32 rounding, rather than an idealized generating polynomial.

Build writes only this family's public cases under triton_eval_cases and its
private records under single_call_vs_tools_challenges/private_data; existing frozen artifacts are protected.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CASES = ("case_48", "case_49")
TOLERANCE = 0.0002
N, DEGREE = 8, 48
SEARCH_SEEDS = tuple(range(501900, 502028))
# The initial arithmetic calibration is also retained in the selection log.
# The threshold was fixed after calibration, before the formal seed sweep.
CALIBRATION_SEEDS = tuple(range(301900, 302028))


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    coefficients = rng.normal(0.0, 1.0, (N, DEGREE + 1)).astype(np.float32)
    anchor = np.float32(1.015625)
    powers = float(anchor) ** np.arange(1, DEGREE + 1)
    coefficients[:, 0] = (
        -np.sum(coefficients[:, 1:].astype(np.float64) * powers[None, :], axis=1)
        + 0.003
    ).astype(np.float32)
    points = (float(anchor) + rng.normal(0.0, 0.00004, N)).astype(np.float32)
    return coefficients, points


def reference(inputs_numpy):
    """FP64 Horner evaluation of supplied FP32 coefficients and points."""
    coefficients, points = (np.asarray(x, dtype=np.float64) for x in inputs_numpy)
    result = coefficients[:, -1].copy()
    for degree in range(coefficients.shape[1] - 2, -1, -1):
        result = result * points + coefficients[:, degree]
    return result


def independent_reference(inputs_numpy):
    """80-digit Decimal direct power sum, independently of the Horner path."""
    coefficients, points = inputs_numpy
    values = []
    with localcontext() as context:
        context.prec = 80
        for row in range(len(points)):
            point = Decimal.from_float(float(points[row]))
            terms = [Decimal.from_float(float(coefficients[row, k])) * point ** k
                     for k in range(coefficients.shape[1])]
            values.append(float(sum(terms, Decimal(0))))
    return np.asarray(values, dtype=np.float64)


def emulate(inputs_numpy):
    """Round the multiply and add separately at every Horner step (no FMA)."""
    coefficients, points = (np.asarray(x, dtype=np.float32) for x in inputs_numpy)
    result = coefficients[:, -1].copy()
    for degree in range(coefficients.shape[1] - 2, -1, -1):
        product = (result * points).astype(np.float32)
        result = (product + coefficients[:, degree]).astype(np.float32)
    return result


def metric(output, reference_output):
    output = np.asarray(output, dtype=np.float64)
    exact = np.asarray(reference_output, dtype=np.float64)
    if output.shape != exact.shape or not np.all(np.isfinite(output)):
        return float("inf")
    numerator = math.sqrt(math.fsum(float(x) ** 2 for x in (output - exact).flat))
    denominator = max(math.sqrt(math.fsum(float(x) ** 2 for x in exact.flat)),
                      0.001 * math.sqrt(exact.size))
    return numerator / denominator


def _candidate(seed):
    inputs = _inputs(seed)
    exact, decimal = reference(inputs), independent_reference(inputs)
    if not np.allclose(exact, decimal, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Independent references disagree for seed {seed}")
    return {
        "seed": seed,
        "error": metric(emulate(inputs), exact),
        "reference_norm": float(np.linalg.norm(exact)),
        "oracle_max_abs_disagreement": float(np.max(np.abs(exact - decimal))),
    }


PROBLEM = """Evaluate eight real polynomials on a fixed public workload.

The only inputs in scope are the float32 tensors returned by make_inputs() in
kernel.py. coefficients has shape (8, 49); points has shape (8,). The polynomial
for row i is exactly defined by the supplied float32 coefficient values:
  reference[i] = sum_{k=0}^{48} coefficients[i,k] * points[i]**k.
Evaluate this reference in float64 using the original supplied float32 inputs.
In particular coefficients[i,0] means the actual stored float32 value, not its
unrounded generating expression. The input generator uses float64 only while
constructing that coefficient, and converts it to float32 before execution.

The output must be a finite vector of shape (8,). Its relative L2 error is
  ||output-reference||_2 / max(||reference||_2, 0.001*sqrt(8)).
The numerical contract requires this error to be <= 0.0002.

Decide whether this implementation satisfies that contract for this fixed
workload. Arbitrary alternative coefficients or points are outside the scope.
The kernel launch disables FP multiply/add fusion, so every Horner multiply
and every Horner addition rounds separately to float32.
"""

KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _polynomial_kernel(Coefficients, Points, Output, N: tl.constexpr,
                       DEGREE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    mask = row < N
    point = tl.load(Points + row, mask=mask, other=0.0).to(tl.float32)
    result = tl.load(Coefficients + row * (DEGREE + 1) + DEGREE,
                     mask=mask, other=0.0).to(tl.float32)
    for step in tl.static_range(0, DEGREE):
        k = DEGREE - 1 - step
        coefficient = tl.load(Coefficients + row * (DEGREE + 1) + k,
                              mask=mask, other=0.0).to(tl.float32)
        product = result * point
        result = product + coefficient
    tl.store(Output + row, result, mask=mask)


def run(coefficients, points):
    output = torch.empty((8,), device=coefficients.device, dtype=torch.float32)
    _polynomial_kernel[(1,)](coefficients, points, output, N=8, DEGREE=48,
                             BLOCK=32, num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    coefficients = rng.normal(0.0, 1.0, (8, 49)).astype(np.float32)
    anchor = np.float32(1.015625)
    powers = float(anchor) ** np.arange(1, 49)
    coefficients[:, 0] = (
        -np.sum(coefficients[:, 1:].astype(np.float64) * powers[None, :], axis=1)
        + 0.003
    ).astype(np.float32)
    points = (float(anchor) + rng.normal(0.0, 0.00004, 8)).astype(np.float32)
    return coefficients, points


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
    selected = {"case_48": min(search, key=lambda row: row["error"]),
                "case_49": max(search, key=lambda row: row["error"])}
    assert selected["case_48"]["error"] <= 0.75 * TOLERANCE
    assert selected["case_49"]["error"] >= 1.25 * TOLERANCE
    answer_key = {"family": "polynomial", "budget": TOLERANCE, "cases": {}}
    for case, selection in selected.items():
        inputs = _inputs(selection["seed"])
        exact, decimal, approximate = reference(inputs), independent_reference(inputs), emulate(inputs)
        error = metric(approximate, exact)
        code = KERNEL_TEMPLATE.replace("__SEED__", str(selection["seed"]))
        folder = root.parent / "triton_eval_cases" / case
        _write_frozen(folder / "kernel.py", code)
        _write_frozen(folder / "problem.txt", PROBLEM)
        _write_frozen(folder / "meta.json", json.dumps(
            {"name": case, "passed": None, "status": "unverified"}, indent=2) + "\n")
        answer_key["cases"][case] = {
            "cpu_ground_truth": "trust" if error <= TOLERANCE else "reject",
            "error": error, "budget": TOLERANCE, "seed": selection["seed"],
            "kernel_sha256": _sha(code.encode()), "problem_sha256": _sha(PROBLEM.encode()),
            "input_sha256": [_sha(x.tobytes()) for x in inputs],
            "reference_norm": float(np.linalg.norm(exact)),
            "reference": exact.tolist(), "independent_reference": decimal.tolist(),
            "emulated_output": approximate.tolist(),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - decimal))),
            "relative_margin_to_budget": abs(error - TOLERANCE) / TOLERANCE,
            "construction_method": "Minimum/maximum error in a fixed 128-seed sweep",
            "mechanism": "Data-dependent conditioning and separate-rounding error of 48-degree Horner evaluation.",
            "floating_point_contract": "FP32 multiply and add separately rounded; enable_fp_fusion=False.",
        }
    _write_frozen(root / "private_data" / "answer_key_polynomial.json", json.dumps(answer_key, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_polynomial.json", json.dumps({
        "family": "polynomial", "numpy_version": np.__version__, "budget": TOLERANCE,
        "threshold_provenance": "0.0002 fixed after arithmetic calibration, before formal seed sweep and any LLM calls.",
        "selection_rule": "Min/max relative error among seeds 501900 through 502027; require >=25% margins.",
        "calibration_candidates": calibration, "candidates": search,
        "selected": {case: row["seed"] for case, row in selected.items()},
        "llm_feedback_used_in_selection": False,
    }, indent=2) + "\n")
    return answer_key


if __name__ == "__main__":
    answer = build(Path(__file__).resolve().parents[3] / "single_call_vs_tools_challenges")
    print(json.dumps({case: {key: row[key] for key in (
        "cpu_ground_truth", "error", "budget", "seed", "oracle_max_abs_disagreement")}
        for case, row in answer["cases"].items()}, indent=2))
