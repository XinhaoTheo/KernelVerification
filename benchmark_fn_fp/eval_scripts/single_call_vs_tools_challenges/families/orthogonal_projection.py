"""Private construction and independent oracles for normalized projection.
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

CASES = ("case_50", "case_51")
TOLERANCE = 0.01
N = 32
EPSILON = 0.00001
CALIBRATION_SEEDS = tuple(range(730100, 730164))
SEARCH_SEEDS = tuple(range(830100, 830356))


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    u = rng.normal(size=N).astype(np.float32)
    b = (1.125 * u.astype(np.float64) + EPSILON * rng.normal(size=N)).astype(np.float32)
    return u, b


def reference(inputs_numpy):
    """FP64 algebraically recentered projection; all input values are FP32."""
    u, b = (np.asarray(x, dtype=np.float64) for x in inputs_numpy)
    # 1.125*u is exact in FP64 for these FP32 inputs. Recentring avoids the
    # cancellation in the direct oracle, while preserving the mathematical map.
    centered = b - 1.125 * u
    denominator = math.fsum(float(x) * float(x) for x in u)
    coefficient = math.fsum(float(x) * float(y) for x, y in zip(u, centered)) / denominator
    residual = centered - coefficient * u
    norm = math.sqrt(math.fsum(float(x) * float(x) for x in residual))
    return residual / norm


def independent_reference(inputs_numpy):
    """80-digit Decimal evaluation of the original uncentered formula."""
    with localcontext() as context:
        context.prec = 80
        u, b = ([Decimal.from_float(float(x)) for x in array] for array in inputs_numpy)
        coefficient = sum((x * y for x, y in zip(u, b)), Decimal(0)) / sum(
            (x * x for x in u), Decimal(0))
        residual = [y - coefficient * x for x, y in zip(u, b)]
        norm = sum((x * x for x in residual), Decimal(0)).sqrt()
        return np.asarray([float(x / norm) for x in residual], dtype=np.float64)


def emulate(inputs_numpy):
    u, b = (np.asarray(x, dtype=np.float32) for x in inputs_numpy)
    numerator, denominator = np.float32(0.0), np.float32(0.0)
    for j in range(N):
        numerator = np.float32(numerator + np.float32(u[j] * b[j]))
        denominator = np.float32(denominator + np.float32(u[j] * u[j]))
    coefficient = np.float32(numerator / denominator)
    residual = (b - (u * coefficient).astype(np.float32)).astype(np.float32)
    norm_squared = np.float32(0.0)
    for j in range(N):
        norm_squared = np.float32(norm_squared + np.float32(residual[j] * residual[j]))
    return (residual / np.sqrt(norm_squared)).astype(np.float32)


def metric(output, reference_output):
    output, exact = (np.asarray(x, dtype=np.float64) for x in (output, reference_output))
    if output.shape != exact.shape or not np.isfinite(output).all():
        return float("inf")
    numerator = math.sqrt(math.fsum(float(x) ** 2 for x in (output - exact).flat))
    denominator = max(math.sqrt(math.fsum(float(x) ** 2 for x in exact.flat)), 1e-12)
    return numerator / denominator


def _candidate(seed):
    inputs = _inputs(seed)
    exact, decimal = reference(inputs), independent_reference(inputs)
    assert np.allclose(exact, decimal, rtol=1e-10, atol=1e-12)
    return {"seed": seed, "error": metric(emulate(inputs), exact),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - decimal)))}


PROBLEM = """Compute a normalized orthogonal projection for a fixed public workload.

Only the two float32 vectors u and b returned by make_inputs() are in scope.
Each has shape (32,). The mathematical reference on these actual stored inputs is
  alpha = sum_j u[j]*b[j] / sum_j u[j]*u[j]
  residual[j] = b[j] - alpha*u[j]
  reference = residual / ||residual||_2.
Use at least float64 accuracy for the reference. Algebraically equivalent
recentring is allowed to avoid numerical cancellation in the reference itself.
The residual on this workload is nonzero.

The output must be a finite vector of shape (32,), and
  ||output-reference||_2 / max(||reference||_2, 1e-12) <= 0.01.

Decide whether this kernel satisfies that contract for the fixed generated
workload. Arbitrary alternative vectors are outside the scope. The kernel uses
float32 arithmetic, sequential accumulations and separately rounded products
and sums; its launch disables FP multiply/add fusion.
"""

KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _projection_kernel(U, B, Output, N: tl.constexpr):
    numerator = tl.full((), 0.0, tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    for j in tl.static_range(0, N):
        u = tl.load(U + j).to(tl.float32)
        b = tl.load(B + j).to(tl.float32)
        numerator = numerator + u * b
        denominator = denominator + u * u
    coefficient = tl.div_rn(numerator, denominator)
    norm_squared = tl.full((), 0.0, tl.float32)
    for j in tl.static_range(0, N):
        u = tl.load(U + j).to(tl.float32)
        b = tl.load(B + j).to(tl.float32)
        residual = b - u * coefficient
        norm_squared = norm_squared + residual * residual
    norm = tl.sqrt(norm_squared)
    j = tl.arange(0, N)
    u = tl.load(U + j).to(tl.float32)
    b = tl.load(B + j).to(tl.float32)
    residual = b - u * coefficient
    tl.store(Output + j, tl.div_rn(residual, norm))


def run(u, b):
    output = torch.empty((32,), device=u.device, dtype=torch.float32)
    _projection_kernel[(1,)](u, b, output, N=32, num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    u = rng.normal(size=32).astype(np.float32)
    b = (1.125 * u.astype(np.float64) + 0.00001 * rng.normal(size=32)).astype(np.float32)
    return u, b


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())
'''


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write_frozen(path, value):
    if path.exists() and path.read_text() != value:
        raise FileExistsError(f"Refusing to replace frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def build(output_root):
    root = Path(output_root)
    calibration = [_candidate(seed) for seed in CALIBRATION_SEEDS]
    candidates = [_candidate(seed) for seed in SEARCH_SEEDS]
    selected = {"case_50": min(candidates, key=lambda row: row["error"]),
                "case_51": max(candidates, key=lambda row: row["error"])}
    assert selected["case_50"]["error"] <= 0.75 * TOLERANCE
    assert selected["case_51"]["error"] >= 1.25 * TOLERANCE
    answer = {"family": "orthogonal_projection", "budget": TOLERANCE, "cases": {}}
    for case, row in selected.items():
        inputs = _inputs(row["seed"])
        exact, decimal, output = reference(inputs), independent_reference(inputs), emulate(inputs)
        code = KERNEL_TEMPLATE.replace("__SEED__", str(row["seed"]))
        folder = root.parent / "triton_eval_cases" / case
        _write_frozen(folder / "kernel.py", code)
        _write_frozen(folder / "problem.txt", PROBLEM)
        _write_frozen(folder / "meta.json", json.dumps(
            {"name": case, "passed": None, "status": "unverified"}, indent=2) + "\n")
        answer["cases"][case] = dict(row, cpu_ground_truth="trust" if row["error"] <= TOLERANCE else "reject",
            budget=TOLERANCE, kernel_sha256=_sha(code.encode()), problem_sha256=_sha(PROBLEM.encode()),
            input_sha256=[_sha(x.tobytes()) for x in inputs], reference=exact.tolist(),
            independent_reference=decimal.tolist(), emulated_output=output.tolist(),
            relative_margin_to_budget=abs(row["error"]-TOLERANCE)/TOLERANCE,
            mechanism="FP32 projection coefficient error rotates a small normalized residual.",
            construction_method="Minimum/maximum error of fixed 256-seed sweep; no model feedback.")
    _write_frozen(root / "private_data" / "answer_key_orthogonal_projection.json", json.dumps(answer, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_orthogonal_projection.json", json.dumps({
        "family": "orthogonal_projection", "numpy_version": np.__version__, "budget": TOLERANCE,
        "threshold_provenance": "0.01 fixed after separate arithmetic calibration and before formal search/model calls.",
        "calibration_candidates": calibration, "candidates": candidates,
        "selection_rule": "Min/max error of seeds 830100..830355; require >=25% threshold margins.",
        "selected": {case: row["seed"] for case, row in selected.items()},
        "llm_feedback_used_in_selection": False,
    }, indent=2) + "\n")
    return answer


if __name__ == "__main__":
    answer = build(Path(__file__).resolve().parents[3] / "single_call_vs_tools_challenges")
    print(json.dumps({case: {k: row[k] for k in ("cpu_ground_truth", "seed", "error", "budget")}
                      for case, row in answer["cases"].items()}, indent=2))
