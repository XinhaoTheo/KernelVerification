"""Private oracle for finite Fourier reconstruction with frequency truncation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CASES = ("case_56", "case_57")
TOLERANCE = 0.15
N, MODES, RETAINED = 64, 16, 6
CALIBRATION_SEEDS = tuple(range(810000, 810032))
SEARCH_SEEDS = tuple(range(811000, 811256))


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    coefficients = rng.normal(0.0, 0.04, (2, MODES)).astype(np.float32)
    coefficients[:, :RETAINED] = rng.normal(0.0, 0.15, (2, RETAINED)).astype(np.float32)
    coefficients[0, 0] = np.float32(coefficients[0, 0] + np.float32(1.0))
    offset = np.asarray([0.25], dtype=np.float32)
    return coefficients, offset


def reference(inputs_numpy):
    """Build the conjugate-symmetric spectrum and use an inverse real FFT."""
    coefficients, offset = (np.asarray(x, dtype=np.float64) for x in inputs_numpy)
    spectrum = np.zeros(N // 2 + 1, dtype=np.complex128)
    spectrum[0] = N * offset[0]
    spectrum[1:MODES + 1] = (N / 2.0) * (coefficients[0] - 1j * coefficients[1])
    return np.fft.irfft(spectrum, n=N)


def independent_reference(inputs_numpy):
    """Direct trigonometric sum with compensated summation, without FFT."""
    coefficients, offset = inputs_numpy
    values = []
    for n in range(N):
        terms = [float(offset[0])]
        for k in range(1, MODES + 1):
            angle = 2.0 * math.pi * k * n / N
            terms.append(float(coefficients[0, k - 1]) * math.cos(angle))
            terms.append(float(coefficients[1, k - 1]) * math.sin(angle))
        values.append(math.fsum(terms))
    return np.asarray(values, dtype=np.float64)


def emulate(inputs_numpy):
    coefficients, offset = (np.asarray(x, dtype=np.float32) for x in inputs_numpy)
    position = np.arange(N, dtype=np.float32)
    output = np.full(N, offset[0], dtype=np.float32)
    for k in range(1, RETAINED + 1):
        angle = position * np.float32(2.0 * math.pi * k / N)
        cosine_term = (coefficients[0, k - 1] * np.cos(angle)).astype(np.float32)
        sine_term = (coefficients[1, k - 1] * np.sin(angle)).astype(np.float32)
        output = (output + cosine_term).astype(np.float32)
        output = (output + sine_term).astype(np.float32)
    return output


def metric(output, reference_output):
    output, exact = np.asarray(output, dtype=np.float64), np.asarray(reference_output, dtype=np.float64)
    if output.shape != exact.shape or not np.all(np.isfinite(output)):
        return float("inf")
    numerator = math.sqrt(math.fsum(float(x) ** 2 for x in (output - exact).flat))
    denominator = max(math.sqrt(math.fsum(float(x) ** 2 for x in exact.flat)), 1e-12)
    return numerator / denominator


def _parseval_error(inputs_numpy):
    coefficients, offset = (np.asarray(x, dtype=np.float64) for x in inputs_numpy)
    omitted_energy = 0.5 * math.fsum(float(x) ** 2 for x in coefficients[:, RETAINED:].flat)
    full_energy = float(offset[0]) ** 2 + 0.5 * math.fsum(float(x) ** 2 for x in coefficients.flat)
    return math.sqrt(omitted_energy / full_energy)


def _candidate(seed):
    inputs = _inputs(seed)
    exact, independent = reference(inputs), independent_reference(inputs)
    if not np.allclose(exact, independent, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Independent references disagree for seed {seed}")
    return {"seed": seed, "error": metric(emulate(inputs), exact),
            "exact_truncation_error_parseval": _parseval_error(inputs),
            "reference_norm": float(np.linalg.norm(exact)),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - independent)))}


PROBLEM = """Reconstruct a real periodic signal from supplied Fourier coefficients.

The only inputs in scope are the float32 tensors returned by make_inputs() in
kernel.py. coefficients has shape (2, 16), and offset has shape (1,). For sample
n = 0,...,63, the mathematical reference uses ALL sixteen supplied modes:
  reference[n] = offset[0] + sum_{k=1}^{16} (
      coefficients[0,k-1] * cos(2*pi*k*n/64)
    + coefficients[1,k-1] * sin(2*pi*k*n/64)).
Evaluate this formula in float64 using the actual supplied float32 values.
The implementation uses a fixed frequency cutoff as an approximation.

Output must be a finite float32 vector of shape (64,). The numerical contract is
  ||output-reference||_2 / max(||reference||_2, 1e-12) <= 0.15.
Decide whether this implementation satisfies that contract for this fixed
public workload. Other coefficient tensors are outside scope.
The launch disables FP multiply/add fusion.
"""

KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct(Coefficients, Offset, Output, N: tl.constexpr,
                 MODES: tl.constexpr, RETAINED: tl.constexpr):
    n = tl.arange(0, N)
    output = tl.full((N,), 0.0, tl.float32) + tl.load(Offset)
    for k in tl.static_range(1, RETAINED + 1):
        a = tl.load(Coefficients + k - 1)
        b = tl.load(Coefficients + MODES + k - 1)
        angle = n.to(tl.float32) * (2.0 * 3.141592653589793 * k / N)
        output = output + a * tl.cos(angle)
        output = output + b * tl.sin(angle)
    tl.store(Output + n, output)


def run(coefficients, offset):
    output = torch.empty((64,), device=coefficients.device, dtype=torch.float32)
    _reconstruct[(1,)](coefficients, offset, output, N=64, MODES=16,
                       RETAINED=6, num_warps=2, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    coefficients = rng.normal(0.0, 0.04, (2, 16)).astype(np.float32)
    coefficients[:, :6] = rng.normal(0.0, 0.15, (2, 6)).astype(np.float32)
    coefficients[0, 0] = np.float32(coefficients[0, 0] + np.float32(1.0))
    offset = np.asarray([0.25], dtype=np.float32)
    return coefficients, offset


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
    selected = {"case_56": min(search, key=lambda row: row["error"]),
                "case_57": max(search, key=lambda row: row["error"])}
    assert selected["case_56"]["error"] <= 0.75 * TOLERANCE
    assert selected["case_57"]["error"] >= 1.25 * TOLERANCE
    answer_key = {"family": "spectral_filter", "budget": TOLERANCE, "cases": {}}
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
            "exact_truncation_error_parseval": _parseval_error(inputs),
            "relative_margin_to_budget": abs(error - TOLERANCE) / TOLERANCE,
            "construction_method": "Minimum/maximum error in a fixed 256-seed sweep",
            "mechanism": "Energy outside a fixed Fourier cutoff relative to total signal energy depends on generated coefficients.",
            "floating_point_contract": "Complete signal defined by supplied FP32 coefficients; retained-mode trigonometric reconstruction uses FP32 without fusion.",
        }
    _write_frozen(root / "private_data" / "answer_key_spectral_filter.json", json.dumps(answer_key, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_spectral_filter.json", json.dumps({
        "family": "spectral_filter", "numpy_version": np.__version__, "budget": TOLERANCE,
        "threshold_provenance": "0.15 fixed in the initial builder before both calibration and formal sweep, and before any LLM calls.",
        "selection_rule": "Min/max relative error among seeds 811000 through 811255; require >=25% margins.",
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
