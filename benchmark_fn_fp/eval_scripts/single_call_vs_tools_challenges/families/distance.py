"""Private oracle for cancellation in distances propagated through RBF regression.
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

CASES = ("case_60", "case_61")
TOLERANCE = 0.05
N, D = 16, 32
CALIBRATION_SEEDS = tuple(range(118100, 118300))
SEARCH_SEEDS = tuple(range(119100, 119356))


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    query = (16.0 + rng.normal(0.0, 0.5, D)).astype(np.float32)
    anchors = (query.astype(np.float64)[None, :] +
               rng.normal(0.0, 0.015625, (N, D))).astype(np.float32)
    values = rng.normal(0.0, 1.0, N).astype(np.float32)
    return query, anchors, values


def reference(inputs_numpy):
    query, anchors, values = (np.asarray(x, dtype=np.float64) for x in inputs_numpy)
    differences = anchors - query[None, :]
    distances = np.sum(differences * differences, axis=1)
    weights = np.exp(-16.0 * distances)
    return np.asarray([np.sum(weights * values) / np.sum(weights)], dtype=np.float64)


def independent_reference(inputs_numpy):
    """80-digit Decimal distance, exponential, and normalized weighted sum."""
    query, anchors, values = inputs_numpy
    with localcontext() as context:
        context.prec = 80
        q = [Decimal.from_float(float(x)) for x in query]
        numerator, denominator = Decimal(0), Decimal(0)
        for row in range(N):
            distance = sum(((Decimal.from_float(float(anchors[row, k])) - q[k]) ** 2
                            for k in range(D)), Decimal(0))
            weight = (-Decimal(16) * distance).exp()
            numerator += weight * Decimal.from_float(float(values[row]))
            denominator += weight
        return np.asarray([float(numerator / denominator)], dtype=np.float64)


def _expanded_distances(inputs_numpy):
    query, anchors, _ = (np.asarray(x, dtype=np.float32) for x in inputs_numpy)
    anchor_norm = np.zeros(N, dtype=np.float32)
    query_norm = np.float32(0)
    dot = np.zeros(N, dtype=np.float32)
    for k in range(D):
        anchor_norm = (anchor_norm + (anchors[:, k] * anchors[:, k]).astype(np.float32)).astype(np.float32)
        query_norm = np.float32(query_norm + np.float32(query[k] * query[k]))
        dot = (dot + (anchors[:, k] * query[k]).astype(np.float32)).astype(np.float32)
    norm_sum = (anchor_norm + query_norm).astype(np.float32)
    twice_dot = (np.float32(2.0) * dot).astype(np.float32)
    return np.maximum((norm_sum - twice_dot).astype(np.float32), np.float32(0))


def emulate(inputs_numpy):
    distances = _expanded_distances(inputs_numpy)
    values = np.asarray(inputs_numpy[2], dtype=np.float32)
    weights = np.exp((np.float32(-16.0) * distances).astype(np.float32)).astype(np.float32)
    weighted_values = (weights * values).astype(np.float32)
    numerator, denominator = np.sum(weighted_values, dtype=np.float32), np.sum(weights, dtype=np.float32)
    return np.asarray([np.float32(numerator / denominator)], dtype=np.float32)


def metric(output, reference_output):
    output, exact = (np.asarray(x, dtype=np.float64) for x in (output, reference_output))
    if output.shape != exact.shape or not np.all(np.isfinite(output)):
        return float("inf")
    return math.sqrt(math.fsum(float(x) ** 2 for x in (output - exact).flat)) / max(
        math.sqrt(math.fsum(float(x) ** 2 for x in exact.flat)), 0.05)


def _candidate(seed):
    inputs = _inputs(seed)
    exact, independent = reference(inputs), independent_reference(inputs)
    if not np.allclose(exact, independent, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Independent references disagree for seed {seed}")
    query, anchors, _ = (np.asarray(x, dtype=np.float64) for x in inputs)
    exact_distances = np.sum((anchors - query[None, :]) ** 2, axis=1)
    return {"seed": seed, "error": metric(emulate(inputs), exact),
            "reference": exact.tolist(),
            "max_distance_absolute_error": float(np.max(np.abs(_expanded_distances(inputs) - exact_distances))),
            "oracle_max_abs_disagreement": float(np.max(np.abs(exact - independent)))}


PROBLEM = """Compute one normalized radial-basis-function regression prediction.

The only workload in scope is the float32 data returned by make_inputs()
in kernel.py: query has shape (32,), anchors has shape (16, 32), and values
has shape (16,). Define the mathematical reference from these actual stored
float32 values, with all the following arithmetic evaluated in float64:
  distance[i] = sum_k (anchors[i,k] - query[k])**2
  weight[i] = exp(-16 * distance[i])
  reference[0] = sum_i weight[i]*values[i] / sum_i weight[i].

The output must be a finite vector of shape (1,). Its numerical error is
  ||output-reference||_2 / max(||reference||_2, 0.05).
The implementation satisfies the contract iff this error is <= 0.05.
The contract concerns the final normalized prediction. It does not impose
separate error requirements on intermediate distances or individual weights.
Alternative inputs and seeds are outside the scope of this fixed workload.

The implementation uses the expanded squared-distance identity and clamps
negative computed distances to zero before the exponential. Its norm and
dot-product accumulators round to float32 each step. FP fusion is disabled.
"""

KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _rbf_kernel(Query, Anchors, Values, Output, N: tl.constexpr, D: tl.constexpr):
    row = tl.arange(0, N)
    anchor_norm = tl.full((N,), 0.0, tl.float32)
    query_norm = tl.full((), 0.0, tl.float32)
    dot = tl.full((N,), 0.0, tl.float32)
    for k in tl.static_range(0, D):
        anchor = tl.load(Anchors + row * D + k).to(tl.float32)
        query = tl.load(Query + k).to(tl.float32)
        anchor_square = anchor * anchor
        query_square = query * query
        cross = anchor * query
        anchor_norm = anchor_norm + anchor_square
        query_norm = query_norm + query_square
        dot = dot + cross
    norm_sum = anchor_norm + query_norm
    twice_dot = 2.0 * dot
    distance = tl.maximum(norm_sum - twice_dot, 0.0)
    weight = tl.exp(-16.0 * distance)
    value = tl.load(Values + row).to(tl.float32)
    numerator = tl.sum(weight * value, 0)
    denominator = tl.sum(weight, 0)
    output = tl.div_rn(numerator, denominator)
    tl.store(Output, output)


def run(query, anchors, values):
    output = torch.empty((1,), device=query.device, dtype=torch.float32)
    _rbf_kernel[(1,)](query, anchors, values, output, N=16, D=32,
                      num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    query = (16.0 + rng.normal(0.0, 0.5, 32)).astype(np.float32)
    anchors = (query.astype(np.float64)[None, :] +
               rng.normal(0.0, 0.015625, (16, 32))).astype(np.float32)
    values = rng.normal(0.0, 1.0, 16).astype(np.float32)
    return query, anchors, values


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
    selected = {"case_60": min(search, key=lambda row: row["error"]),
                "case_61": max(search, key=lambda row: row["error"])}
    assert selected["case_60"]["error"] <= 0.75 * TOLERANCE
    assert selected["case_61"]["error"] >= 1.25 * TOLERANCE
    answer_key = {"family": "distance", "budget": TOLERANCE, "cases": {}}
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
            "max_distance_absolute_error": selection["max_distance_absolute_error"],
            "construction_method": "Minimum/maximum error in a fixed 256-seed sweep",
            "mechanism": "Cancellation in norm-expanded squared distances perturbs RBF weights; signed target aggregation can cancel or amplify the final prediction error.",
            "floating_point_contract": "Sequential FP32 norm/dot sums, separate operations, final FP32 normalized weighted sum.",
        }
    _write_frozen(root / "private_data" / "answer_key_distance.json", json.dumps(answer_key, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_distance.json", json.dumps({
        "family": "distance", "numpy_version": np.__version__, "budget": TOLERANCE,
        "threshold_provenance": "0.05 fixed after arithmetic calibration of offsets 8,16,32 and before formal seed sweep or any model calls. The retained calibration uses the selected offset 16.",
        "selection_rule": "Min/max relative prediction error among seeds 119100 through 119355; require >=25% margins.",
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
