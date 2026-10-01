"""Private oracle: midpoint integration of oscillatory, public input functions.
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

CASES = ("case_54", "case_55")
TOLERANCE = 0.035
ROWS, TERMS, GRID = 4, 8, 32
CALIBRATION_SEEDS = tuple(range(710000, 710032))
SEARCH_SEEDS = tuple(range(711000, 711256))
DECIMAL_PI = Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679")


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    amplitudes = rng.normal(0.0, 0.1, (ROWS, TERMS)).astype(np.float32)
    frequencies = rng.uniform(0.5, 220.0, (ROWS, TERMS)).astype(np.float32)
    phases = rng.uniform(-math.pi, math.pi, (ROWS, TERMS)).astype(np.float32)
    return amplitudes, frequencies, phases


def reference(inputs_numpy):
    amplitudes, frequencies, phases = (np.asarray(x, dtype=np.float64) for x in inputs_numpy)
    integrals = amplitudes * (np.cos(phases) - np.cos(phases + frequencies)) / frequencies
    return 1.0 + np.sum(integrals, axis=1)


def _decimal_cos(value):
    period = 2 * DECIMAL_PI
    value %= period
    if value > DECIMAL_PI:
        value -= period
    if value < -DECIMAL_PI:
        value += period
    total = term = Decimal(1)
    for k in range(1, 160):
        term *= -(value * value) / Decimal((2 * k - 1) * (2 * k))
        updated = total + term
        if updated == total:
            return total
        total = updated
    raise ArithmeticError("Decimal cosine did not converge")


def independent_reference(inputs_numpy):
    amplitudes, frequencies, phases = inputs_numpy
    values = []
    with localcontext() as context:
        context.prec = 80
        for row in range(ROWS):
            total = Decimal(1)
            for k in range(TERMS):
                a, w, p = [Decimal.from_float(float(x[row, k]))
                           for x in (amplitudes, frequencies, phases)]
                total += a * (_decimal_cos(p) - _decimal_cos(p + w)) / w
            values.append(float(total))
    return np.asarray(values, dtype=np.float64)


def emulate(inputs_numpy):
    amplitudes, frequencies, phases = (np.asarray(x, dtype=np.float32) for x in inputs_numpy)
    points = (np.arange(GRID, dtype=np.float32) + np.float32(0.5)) / np.float32(GRID)
    samples = np.ones((ROWS, GRID), dtype=np.float32)
    for k in range(TERMS):
        angle = (frequencies[:, k, None] * points[None, :]).astype(np.float32)
        angle = (angle + phases[:, k, None]).astype(np.float32)
        term = (amplitudes[:, k, None] * np.sin(angle)).astype(np.float32)
        samples = (samples + term).astype(np.float32)
    return (np.sum(samples, axis=1, dtype=np.float32) / np.float32(GRID)).astype(np.float32)


def metric(output, reference_output):
    output, exact = np.asarray(output, dtype=np.float64), np.asarray(reference_output, dtype=np.float64)
    if output.shape != exact.shape or not np.all(np.isfinite(output)):
        return float("inf")
    numerator = math.sqrt(math.fsum(float(x) ** 2 for x in (output - exact).flat))
    denominator = max(math.sqrt(math.fsum(float(x) ** 2 for x in exact.flat)), 1e-12)
    return numerator / denominator


def _candidate(seed):
    inputs = _inputs(seed)
    exact, independent = reference(inputs), independent_reference(inputs)
    if not np.allclose(exact, independent, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Independent references disagree for seed {seed}")
    return {"seed": seed, "error": metric(emulate(inputs), exact),
            "reference_norm": float(np.linalg.norm(exact)),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - independent)))}


PROBLEM = """Integrate four oscillatory real functions over the interval [0, 1].

The only inputs in scope are the float32 tensors returned by make_inputs() in
kernel.py. amplitudes, frequencies, and phases each have shape (4, 8). The exact
real function for row r is defined by the supplied float32 values:
  f_r(t) = 1 + sum_{k=0}^{7} amplitudes[r,k] * sin(frequencies[r,k]*t + phases[r,k]).
The mathematical reference is integral_0^1 f_r(t) dt, evaluated in float64:
  reference[r] = 1 + sum_k amplitudes[r,k] *
      (cos(phases[r,k])-cos(phases[r,k]+frequencies[r,k])) / frequencies[r,k].
This is the continuous integral; the 32-point grid in the implementation is an
approximation method, not the definition of the reference.

Output must be a finite float32 vector of shape (4,). The numerical contract is
  ||output-reference||_2 / max(||reference||_2, 1e-12) <= 0.035.
Decide whether this implementation satisfies that contract for this fixed
public workload. Other amplitudes, frequencies, or phases are outside scope.
The launch disables FP multiply/add fusion.
"""

KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _integrate(Amplitudes, Frequencies, Phases, Output,
               TERMS: tl.constexpr, GRID: tl.constexpr):
    row = tl.program_id(0)
    cell = tl.arange(0, GRID)
    point = (cell.to(tl.float32) + 0.5) / GRID
    value = tl.full((GRID,), 1.0, tl.float32)
    for k in tl.static_range(0, TERMS):
        amplitude = tl.load(Amplitudes + row * TERMS + k)
        frequency = tl.load(Frequencies + row * TERMS + k)
        phase = tl.load(Phases + row * TERMS + k)
        angle = frequency * point
        angle = angle + phase
        value = value + amplitude * tl.sin(angle)
    estimate = tl.sum(value, axis=0) / GRID
    tl.store(Output + row, estimate)


def run(amplitudes, frequencies, phases):
    output = torch.empty((4,), device=amplitudes.device, dtype=torch.float32)
    _integrate[(4,)](amplitudes, frequencies, phases, output,
                     TERMS=8, GRID=32, num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    amplitudes = rng.normal(0.0, 0.1, (4, 8)).astype(np.float32)
    frequencies = rng.uniform(0.5, 220.0, (4, 8)).astype(np.float32)
    phases = rng.uniform(-np.pi, np.pi, (4, 8)).astype(np.float32)
    return amplitudes, frequencies, phases


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
    selected = {"case_54": min(search, key=lambda row: row["error"]),
                "case_55": max(search, key=lambda row: row["error"])}
    assert selected["case_54"]["error"] <= 0.75 * TOLERANCE
    assert selected["case_55"]["error"] >= 1.25 * TOLERANCE
    answer_key = {"family": "quadrature", "budget": TOLERANCE, "cases": {}}
    for case, selection in selected.items():
        inputs = _inputs(selection["seed"])
        exact, independent, approximate = reference(inputs), independent_reference(inputs), emulate(inputs)
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
            "reference": exact.tolist(), "independent_reference": independent.tolist(),
            "emulated_output": approximate.tolist(),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - independent))),
            "relative_margin_to_budget": abs(error - TOLERANCE) / TOLERANCE,
            "construction_method": "Minimum/maximum error in a fixed 256-seed sweep",
            "mechanism": "Data-dependent quadrature aliasing of oscillatory integrands on a fixed midpoint grid.",
            "floating_point_contract": "Actual supplied FP32 coefficients define continuous functions; FP32 midpoint arithmetic with fusion disabled.",
        }
    _write_frozen(root / "private_data" / "answer_key_quadrature.json", json.dumps(answer_key, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_quadrature.json", json.dumps({
        "family": "quadrature", "numpy_version": np.__version__, "budget": TOLERANCE,
        "threshold_provenance": "0.035 fixed in the initial builder before both calibration and formal sweep, and before any LLM calls.",
        "selection_rule": "Min/max relative error among seeds 711000 through 711255; require >=25% margins.",
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
