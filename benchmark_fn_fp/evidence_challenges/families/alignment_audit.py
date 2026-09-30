"""Private fixed-workload oracles for shared latent gauge compatibility.

Only the generated eval_cases files are visible to benchmark agents.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

CASES = ("case_68", "case_69")
V, N, D = 2, 8, 4
TOLERANCE = 0.05
SEARCH_SEEDS = range(171200, 171456)
PERMUTATIONS = tuple(itertools.permutations(range(D)))
SIGNS = tuple(itertools.product((-1, 1), repeat=D))
TRANSFORMS = tuple((p, s) for p in PERMUTATIONS for s in SIGNS)


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    features = rng.normal(0.0, 1.0, (V, N, D)).astype(np.float32)
    features[0, :, 2:] = 0
    features[1, :, (0, 3)] = 0
    anchors = rng.normal(0.0, 1.0, (V, D)).astype(np.float32)
    return features, anchors


def emulate(inputs_numpy):
    features, anchors = inputs_numpy
    output = np.zeros_like(features)
    for view in range(V):
        if len(set(float(v) for v in np.abs(anchors[view]))) != D:
            raise ValueError("Anchor rank tie in fixed workload")
        for column in range(D):
            slot = int(np.count_nonzero(np.abs(anchors[view]) < abs(anchors[view, column])))
            output[view, :, slot] = features[view, :, column] * (
                np.float32(1) if anchors[view, column] >= 0 else np.float32(-1))
    return output


def reference(output, features):
    """Exhaustively compare all 24 permutations times 16 sign vectors."""
    target = np.asarray(features, np.float64)
    actual = np.asarray(output, np.float64)
    transformed = np.stack([target[..., p] * np.asarray(s) for p, s in TRANSFORMS])
    squared = np.sum((transformed - actual) ** 2, axis=tuple(range(1, transformed.ndim)))
    index = int(np.argmin(squared))
    denominator = max(float(np.linalg.norm(target)), 1e-12)
    return {"error": math.sqrt(float(squared[index])) / denominator,
            "closest_reference": transformed[index], "transform": TRANSFORMS[index],
            "minimizer_count": int(np.count_nonzero(np.isclose(squared, squared[index], atol=1e-12, rtol=0)))}


def independent_reference(output, features):
    """Signed bipartite assignment costs accumulated independently by fsum.

    An edge links one original channel to one output slot. A single sign on
    that edge must explain all rows of every view; signs cannot vary by view.
    Enumerating the 24 assignments then minimizing each edge's two signs is
    independent of constructing and comparing the 384 full transformed arrays.
    """
    x = np.asarray(features, np.float64).reshape(-1, D)
    y = np.asarray(output, np.float64).reshape(-1, D)
    costs = {}
    for original in range(D):
        for slot in range(D):
            costs[original, slot] = {
                sign: math.fsum((float(y[r, slot]) - sign * float(x[r, original])) ** 2
                                for r in range(x.shape[0]))
                for sign in (-1, 1)}
    best = None
    for permutation in PERMUTATIONS:
        signs = tuple(min((-1, 1), key=lambda s: costs[original, slot][s])
                      for slot, original in enumerate(permutation))
        cost = math.fsum(costs[original, slot][signs[slot]]
                         for slot, original in enumerate(permutation))
        item = (cost, permutation, signs)
        if best is None or item < best:
            best = item
    denominator = max(math.sqrt(math.fsum(float(v) ** 2 for v in x.flat)), 1e-12)
    return {"error": math.sqrt(best[0]) / denominator,
            "closest_reference": np.asarray(features, np.float64)[..., best[1]] * np.asarray(best[2]),
            "transform": (best[1], best[2])}


PROBLEM = """Audit a channel canonicalizer for a shared multi-view latent space.

Two sensor views use the same four latent channels. A view can have an
unobserved channel: a zero column means that feature is absent from that
view, not that its identity is a new independent channel. The fixed workload
is exactly the stored float32 arrays returned by make_inputs() in kernel.py:
features has shape (2, 8, 4), and anchors has shape (2, 4). Anchors are
per-view summaries used by this candidate canonicalizer; they do not change
the mathematical target or the permitted representation equivalence.

The mathematical target is the input features tensor itself, up to ONE
shared signed permutation of its four latent columns. Precisely, define
  E = min over p in permutations(0,1,2,3), s in {-1,+1}^4 of
      ||output[v,r,k] - s[k] * features[v,r,p[k]]||_F
      / max(||features||_F, 1e-12).
The numerator and denominator include BOTH views and ALL eight rows.
The same p and s must be used for every view and every row. There are exactly
384 permitted transforms; arbitrary rotations, row-dependent transforms,
and view-dependent transforms are outside this equivalence. Missing columns
can make a view's best-fitting transform nonunique. No particular canonical
column order or sign is required as long as one shared transform fits.

run(features, anchors) must return finite float32 output of shape (2, 8, 4)
without modifying either input. It satisfies this fixed workload iff E <=
0.05, computed in float64 or more accurately from the actual float32 inputs
and output. Other seeds and arrays are outside this contract.

kernel.py includes initial_probe(), an executable prior experiment. Its code
and observed output are evidence to assess, not an authoritative verdict or
a separate specification. All evaluation systems receive the same material.
"""


KERNEL_TEMPLATE = '''import itertools
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _canonicalize(Features, Anchors, Out):
    view = tl.program_id(0)
    column = tl.program_id(1)
    anchor = tl.load(Anchors + view * 4 + column)
    a0 = tl.abs(tl.load(Anchors + view * 4))
    a1 = tl.abs(tl.load(Anchors + view * 4 + 1))
    a2 = tl.abs(tl.load(Anchors + view * 4 + 2))
    a3 = tl.abs(tl.load(Anchors + view * 4 + 3))
    magnitude = tl.abs(anchor)
    slot = ((a0 < magnitude).to(tl.int32) + (a1 < magnitude).to(tl.int32)
            + (a2 < magnitude).to(tl.int32) + (a3 < magnitude).to(tl.int32))
    rows = tl.arange(0, 8)
    values = tl.load(Features + view * 32 + rows * 4 + column)
    sign = tl.where(anchor >= 0, 1.0, -1.0)
    tl.store(Out + view * 32 + rows * 4 + slot, values * sign)


def run(features, anchors):
    output = torch.empty_like(features)
    _canonicalize[(2, 4)](features, anchors, output, num_warps=1)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(__SEED__))
    features = rng.normal(0.0, 1.0, (2, 8, 4)).astype(np.float32)
    features[0, :, 2:] = 0
    features[1, :, (0, 3)] = 0
    anchors = rng.normal(0.0, 1.0, (2, 4)).astype(np.float32)
    return features, anchors


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())


def initial_probe():
    features, anchors = make_inputs()
    output = run(features, anchors)
    x = features.detach().cpu().numpy().astype(np.float64)
    y = output.detach().cpu().numpy().astype(np.float64)
    local = []
    for view in range(2):
        best_error = float("inf")
        best_transform = None
        for permutation in itertools.permutations(range(4)):
            for signs in itertools.product((-1, 1), repeat=4):
                expected = x[view][:, permutation] * np.asarray(signs)
                error = float(np.linalg.norm(y[view] - expected) /
                              max(np.linalg.norm(x[view]), 1e-12))
                if error < best_error:
                    best_error = error
                    best_transform = {"permutation": list(permutation), "signs": list(signs)}
        local.append({"view": view, "relative_error": best_error,
                      "chosen_transform": best_transform})
    gram_error = float(np.linalg.norm(y @ y.transpose(0, 2, 1) -
                                      x @ x.transpose(0, 2, 1)))
    return {"per_view_alignment": local, "within_view_gram_error": gram_error,
            "tolerance": 0.05,
            "passed": bool(all(row["relative_error"] <= 0.05 for row in local))}
'''


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_frozen(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise FileExistsError(f"Refusing to overwrite frozen artifact {path}")
    path.write_text(content)


def build(root):
    root = Path(root)
    candidates = []
    for seed in SEARCH_SEEDS:
        inputs = _inputs(seed)
        output = emulate(inputs)
        oracle, independent = reference(output, inputs[0]), independent_reference(output, inputs[0])
        disagreement = abs(oracle["error"] - independent["error"])
        if disagreement > 1e-12:
            raise ValueError(f"Independent oracle mismatch for {seed}")
        local = [reference(output[v], inputs[0][v]) for v in range(V)]
        candidates.append({"seed": seed, "error": oracle["error"],
            "closest_shared_transform": oracle["transform"],
            "shared_minimizer_count": oracle["minimizer_count"],
            "oracle_error_disagreement": disagreement,
            "initial_per_view_errors": [row["error"] for row in local],
            "local_representative_transforms": [row["transform"] for row in local],
            "local_minimizer_counts": [row["minimizer_count"] for row in local],
            "local_representatives_differ": local[0]["transform"] != local[1]["transform"]})
    eligible_pass = [r for r in candidates if r["error"] <= TOLERANCE / 2 and r["local_representatives_differ"]]
    eligible_fail = [r for r in candidates if r["error"] >= 2 * TOLERANCE]
    if not eligible_pass or not eligible_fail:
        raise ValueError("Fixed CPU sweep did not yield the specified pair")
    selected = {CASES[0]: eligible_pass[0], CASES[1]: eligible_fail[0]}
    answers = {"family": "alignment_audit", "budget": TOLERANCE, "cases": {}}
    for name, row in selected.items():
        code = KERNEL_TEMPLATE.replace("__SEED__", str(row["seed"]))
        folder = root / "eval_cases" / name
        _write_frozen(folder / "kernel.py", code)
        _write_frozen(folder / "problem.txt", PROBLEM)
        _write_frozen(folder / "meta.json", json.dumps({"name": name, "passed": None, "status": "unverified"}, indent=2) + "\n")
        arrays = _inputs(row["seed"])
        answers["cases"][name] = {**row,
            "cpu_ground_truth": "trust" if row["error"] <= TOLERANCE else "reject",
            "budget": TOLERANCE, "kernel_sha256": _sha(code.encode()),
            "problem_sha256": _sha(PROBLEM.encode()), "input_sha256": [_sha(a.tobytes()) for a in arrays],
            "mechanism": "Per-view gauge alignment does not establish existence of a single shared gauge; nonunique local solutions also mean different local representatives do not prove incompatibility."}
    _write_frozen(root / "private_data" / "answer_key_alignment.json", json.dumps(answers, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_alignment.json", json.dumps({
        "family": "alignment_audit", "numpy_version": np.__version__, "budget": TOLERANCE,
        "candidates": candidates, "selected": {name: row["seed"] for name, row in selected.items()},
        "selection_rule": "First seed passing at <= half tolerance whose independent per-view first minimizers differ, and first seed failing at >= twice tolerance; every seed in fixed range recorded.",
        "threshold_provenance": "0.05 chosen before the sole 256-seed CPU search, GPU validation, and any model calls; no calibration or LLM feedback.",
        "llm_feedback_used_in_selection": False}, indent=2) + "\n")
    return answers


def validate_case(name, kernel):
    import torch

    if name not in CASES:
        raise ValueError(name)
    arrays = kernel.make_inputs_numpy()
    hashes = [_sha(a.tobytes()) for a in arrays]
    outputs = []
    for _ in range(10):
        inputs = kernel.make_inputs()
        if [_sha(a.detach().cpu().numpy().tobytes()) for a in inputs] != hashes:
            raise ValueError("Actual GPU inputs differ from canonical byte hashes")
        if any(a.dtype != torch.float32 or tuple(a.shape) != expected.shape
               for a, expected in zip(inputs, arrays)):
            raise ValueError("Actual GPU input shape or dtype differs")
        result = kernel.run(*inputs)
        if result.dtype != torch.float32 or tuple(result.shape) != (V, N, D):
            raise ValueError("GPU output violates dtype or shape")
        output = result.detach().cpu().numpy().copy()
        if not np.all(np.isfinite(output)):
            raise ValueError("GPU output is nonfinite")
        if [_sha(a.detach().cpu().numpy().tobytes()) for a in inputs] != hashes:
            raise ValueError("Kernel modified inputs")
        outputs.append(output)
    torch.cuda.synchronize()
    if any(not np.array_equal(outputs[0], output) for output in outputs[1:]):
        raise ValueError("GPU output is not repeatable")
    output = outputs[0]
    emulated = emulate(arrays)
    oracle, independent = reference(output, arrays[0]), independent_reference(output, arrays[0])
    disagreement = abs(oracle["error"] - independent["error"])
    if disagreement > 1e-12 or not np.array_equal(output, emulated):
        raise ValueError("Independent oracle or GPU/CPU mismatch")
    return {"case": name, "ground_truth": "trust" if oracle["error"] <= TOLERANCE else "reject",
        "error": oracle["error"], "budget": TOLERANCE, "tolerance": TOLERANCE,
        "output": output.tolist(), "reference": oracle["closest_reference"].tolist(),
        "independent_reference": independent["closest_reference"].tolist(),
        "closest_shared_transform": oracle["transform"],
        "shared_minimizer_count": oracle["minimizer_count"],
        "independent_error": independent["error"], "oracle_error_disagreement": disagreement,
        "cpu_emulated_error": reference(emulated, arrays[0])["error"],
        "gpu_cpu_emulation_max_abs_disagreement": float(np.max(np.abs(output - emulated))),
        "input_sha256": hashes, "actual_gpu_inputs_match_canonical": True,
        "gpu_inputs_unchanged": True, "output_contract_shape_dtype_finite": True,
        "output_sha256": _sha(output.tobytes()), "repeat_count": 10, "repeatable": True,
        "output_shape": list(output.shape), "output_dtype": str(output.dtype),
        "initial_probe": kernel.initial_probe()}


if __name__ == "__main__":
    answer = build(Path(__file__).resolve().parents[1])
    print(json.dumps({name: {key: row[key] for key in ("seed", "cpu_ground_truth", "error", "budget")}
                      for name, row in answer["cases"].items()}, indent=2))
