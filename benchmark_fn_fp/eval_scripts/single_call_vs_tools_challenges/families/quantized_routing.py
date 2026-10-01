"""Private deterministic nearest-neighbour routing construction and oracles.
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

CASES = ("case_52", "case_53")
TOLERANCE = 0.1
N, D, E = 16, 32, 8
SEARCH_SEEDS = tuple(range(840100, 840356))


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    query = rng.normal(0.0, 0.3, D).astype(np.float32)
    offsets = rng.normal(size=(N, D))
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
    radii = 1.0 + rng.uniform(-0.002, 0.002, N)
    candidates = (query.astype(np.float64)[None, :] + offsets * radii[:, None]).astype(np.float32)
    embeddings = rng.normal(size=(N, E)).astype(np.float32)
    return query, candidates, embeddings


def _distances(inputs_numpy):
    query, candidates, _ = inputs_numpy
    return np.sum((candidates.astype(np.float64) - query.astype(np.float64)[None, :]) ** 2, axis=1)


def _quantized_distances(inputs_numpy):
    query, candidates, _ = (np.asarray(x, dtype=np.float32) for x in inputs_numpy)
    query_q = (np.floor(query * np.float32(8.0) + np.float32(0.5)) * np.float32(0.125)).astype(np.float32)
    candidates_q = (np.floor(candidates * np.float32(8.0) + np.float32(0.5)) * np.float32(0.125)).astype(np.float32)
    distances = np.zeros(N, dtype=np.float32)
    for j in range(D):
        delta = (candidates_q[:, j] - query_q[j]).astype(np.float32)
        distances = (distances + (delta * delta).astype(np.float32)).astype(np.float32)
    return distances


def reference(inputs_numpy):
    winner = int(np.argmin(_distances(inputs_numpy)))
    return inputs_numpy[2][winner].astype(np.float64)


def independent_reference(inputs_numpy):
    """Decimal80 direct squared distances and exact lowest-index tie rule."""
    query, candidates, embeddings = inputs_numpy
    with localcontext() as context:
        context.prec = 80
        q = [Decimal.from_float(float(x)) for x in query]
        distances = []
        for row in candidates:
            distances.append(sum(((Decimal.from_float(float(x)) - y) ** 2 for x, y in zip(row, q)), Decimal(0)))
        winner = min(range(N), key=lambda i: (distances[i], i))
    return embeddings[winner].astype(np.float64)


def emulate(inputs_numpy):
    winner = int(np.argmin(_quantized_distances(inputs_numpy)))
    return inputs_numpy[2][winner].copy()


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
    assert np.array_equal(exact, decimal)
    distances, quantized = _distances(inputs), _quantized_distances(inputs)
    return {"seed": seed, "error": metric(emulate(inputs), exact),
            "reference_winner": int(np.argmin(distances)),
            "quantized_winner": int(np.argmin(quantized)),
            "reference_top2_gap": float(np.sort(distances)[1] - np.sort(distances)[0]),
            "quantized_top2_gap": float(np.sort(quantized)[1] - np.sort(quantized)[0]),
            "oracle_max_abs_disagreement": 0.0}


PROBLEM = """Select a nearest candidate and return its embedding on a fixed workload.

Only make_inputs() supplies the inputs in scope: query is float32 of shape
(32,), candidates is float32 of shape (16, 32), embeddings is float32 of shape
(16, 8). For the reference, use the original supplied float32 coordinates
without quantization. Compute squared Euclidean distances in float64:
  d[i] = sum_j (candidates[i,j] - query[j])**2.
Select the index with smallest d[i], breaking ties in favour of the smallest
index. The reference output is embeddings[index, :].

The kernel returns a finite vector of shape (8,). It satisfies the contract iff
  ||output-reference||_2 / max(||reference||_2, 1e-12) <= 0.1.

Decide whether this implementation meets the numerical contract for this fixed
generated workload. Alternative queries, candidates and embeddings are outside
the scope. The kernel's approximate route quantizes coordinates to a grid
using floor(x*8+0.5)/8, then uses float32 squared distances and the same tie rule.
The final embedding values are gathered without quantization.
"""

KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(Query, Candidates, Embeddings, Output, N: tl.constexpr,
                  D: tl.constexpr, E: tl.constexpr):
    row = tl.arange(0, N)
    distances = tl.full((N,), 0.0, tl.float32)
    for j in tl.static_range(0, D):
        query = tl.load(Query + j).to(tl.float32)
        candidate = tl.load(Candidates + row * D + j).to(tl.float32)
        query_q = tl.floor(query * 8.0 + 0.5) * 0.125
        candidate_q = tl.floor(candidate * 8.0 + 0.5) * 0.125
        delta = candidate_q - query_q
        distances = distances + delta * delta
    minimum = tl.min(distances, axis=0)
    winner = tl.min(tl.where(distances == minimum, row, 2147483647), axis=0)
    component = tl.arange(0, E)
    result = tl.load(Embeddings + winner * E + component)
    tl.store(Output + component, result)


def run(query, candidates, embeddings):
    output = torch.empty((8,), device=query.device, dtype=torch.float32)
    _route_kernel[(1,)](query, candidates, embeddings, output, N=16, D=32, E=8,
                        num_warps=1, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    query = rng.normal(0.0, 0.3, 32).astype(np.float32)
    offsets = rng.normal(size=(16, 32))
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
    radii = 1.0 + rng.uniform(-0.002, 0.002, 16)
    candidates = (query.astype(np.float64)[None, :] + offsets * radii[:, None]).astype(np.float32)
    embeddings = rng.normal(size=(16, 8)).astype(np.float32)
    return query, candidates, embeddings


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
    candidates = [_candidate(seed) for seed in SEARCH_SEEDS]
    eligible = [row for row in candidates if row["reference_top2_gap"] > 1e-5 and row["quantized_top2_gap"] >= 0.015625]
    selected = {"case_52": next(row for row in eligible if row["error"] <= 0.75 * TOLERANCE),
                "case_53": next(row for row in eligible if row["error"] >= 1.25 * TOLERANCE)}
    answer = {"family": "quantized_routing", "budget": TOLERANCE, "cases": {}}
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
            reference_distances=_distances(inputs).tolist(),
            quantized_distances=_quantized_distances(inputs).tolist(),
            relative_margin_to_budget=abs(row["error"]-TOLERANCE)/TOLERANCE,
            mechanism="Coordinate quantization may change the nearest-neighbour route and gathered embedding.",
            construction_method="First robust pass/fail in fixed 256-seed sweep, without model feedback.")
    _write_frozen(root / "private_data" / "answer_key_quantized_routing.json", json.dumps(answer, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_log_quantized_routing.json", json.dumps({
        "family": "quantized_routing", "numpy_version": np.__version__, "budget": TOLERANCE,
        "threshold_provenance": "0.1 fixed before seed search or model calls; accepted routes reproduce the embedding exactly.",
        "candidates": candidates,
        "selection_rule": "First pass/fail among seeds 840100..840355 with exact top-2 gap >1e-5 and quantized gap >=1/64.",
        "selected": {case: row["seed"] for case, row in selected.items()},
        "llm_feedback_used_in_selection": False,
    }, indent=2) + "\n")
    return answer


if __name__ == "__main__":
    answer = build(Path(__file__).resolve().parents[3] / "single_call_vs_tools_challenges")
    print(json.dumps({case: {k: row[k] for k in ("cpu_ground_truth", "seed", "error", "budget")}
                      for case, row in answer["cases"].items()}, indent=2))
