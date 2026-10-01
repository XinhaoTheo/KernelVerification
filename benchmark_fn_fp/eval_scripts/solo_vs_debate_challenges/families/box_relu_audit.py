"""Private continuous-box certificate for pruning a positive ReLU residual.

Vertices give a complete convex-domain maximum. An independent exact integer
active-subset support-function calculation certifies that same maximum.

Build writes only this family's public cases under triton_eval_cases and its
private records under solo_vs_debate_challenges/private_data; existing frozen artifacts are protected.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

CASES = ("case_72", "case_73")
D, M = 12, 6
TOLERANCE = 1.0
BASE_ROUNDOFF_BOUND = 1e-7
SEARCH_SEEDS = range(194000, 194064)
VERTICES = np.asarray(list(itertools.product((-1, 1), repeat=D)), dtype=np.float32)
SUBSETS = np.asarray(list(itertools.product((0, 1), repeat=M)), dtype=np.int64)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    center = rng.choice(np.asarray([-1, 1]), size=D)
    flip_probability = rng.uniform(0.05, 0.4)
    flips = np.where(rng.uniform(size=(M, D)) < flip_probability, -1, 1)
    magnitudes = rng.choice(np.asarray([0.5, 1.0]), size=(M, D))
    weights = (center[None, :] * flips * magnitudes).astype(np.float32)
    biases = (0.75 * np.abs(weights).sum(axis=1)).astype(np.float32)
    coefficients = np.full(M, 0.25, dtype=np.float32)
    smoke = np.concatenate((np.zeros((1, D)), np.eye(D), -np.eye(D),
                            rng.uniform(-1.0, 1.0, size=(16, D))), axis=0).astype(np.float32)
    return smoke, weights, biases, coefficients


def residual(x, weights, biases, coefficients):
    """FP64 vertex calculation; all vertex terms are exact dyadic values."""
    products = np.einsum("nd,jd->nj", np.asarray(x, np.float64),
                         np.asarray(weights, np.float64), optimize=False)
    return np.sum(np.maximum(products - np.asarray(biases, np.float64), 0)
                  * np.asarray(coefficients, np.float64), axis=1)


def base(x):
    x = np.asarray(x, np.float64)
    return 0.25 * x[:, 0] + 0.5 * x[:, 1]


def emulate(x):
    x = np.asarray(x, np.float32)
    return np.asarray(np.float32(0.25) * x[:, 0] + np.float32(0.5) * x[:, 1], np.float32)


def independent_certificate(weights, biases, coefficients):
    """Exact integer support functions on all 64 active neuron subsets.

    sum_j c_j ReLU(w_j*x-b_j) = max_S sum_{j in S} c_j(w_j*x-b_j).
    A linear form's maximum on [-1,1]^D is its L1 coefficient norm.
    All weighted parameters use denominator 32 in this construction.
    """
    scaled_w = np.asarray(weights, np.float64) * np.asarray(coefficients, np.float64)[:, None] * 32
    scaled_b = np.asarray(biases, np.float64) * np.asarray(coefficients, np.float64) * 32
    if not np.array_equal(scaled_w, np.rint(scaled_w)) or not np.array_equal(scaled_b, np.rint(scaled_b)):
        raise ValueError("Parameters are not exactly on the declared dyadic grid")
    wi, bi = scaled_w.astype(np.int64), scaled_b.astype(np.int64)
    combined = SUBSETS @ wi
    support = np.abs(combined).sum(axis=1) - SUBSETS @ bi
    winner = int(np.argmax(support))
    witness = np.where(combined[winner] < 0, -1.0, 1.0).astype(np.float32)
    return {"maximum": int(support[winner]) / 32,
            "maximum_numerator": int(support[winner]), "denominator": 32,
            "active_subset": SUBSETS[winner].tolist(), "witness": witness.tolist(),
            "subset_support_numerators": support.tolist()}


def _candidate(seed):
    smoke, weights, biases, coefficients = _inputs(seed)
    values = residual(VERTICES, weights, biases, coefficients)
    index = int(np.argmax(values))
    maximum = float(values[index])
    certificate = independent_certificate(weights, biases, coefficients)
    if maximum != certificate["maximum"]:
        raise ValueError(f"Independent complete-domain certificates disagree: {seed}")
    if float(residual(np.asarray([certificate["witness"]]), weights, biases, coefficients)[0]) != maximum:
        raise ValueError("Independent certificate's witness does not reach its maximum")
    smoke_errors = np.abs(emulate(smoke).astype(np.float64) - base(smoke)
                          - residual(smoke, weights, biases, coefficients))
    individual = coefficients.astype(np.float64) * np.maximum(
        np.abs(weights.astype(np.float64)).sum(axis=1) - biases, 0)
    return {"seed": seed, "error": maximum, "tolerance": TOLERANCE,
        "cpu_ground_truth": ("trust" if maximum + BASE_ROUNDOFF_BOUND <= TOLERANCE
                             else "reject" if maximum > TOLERANCE else "requires_boundary_analysis"),
        "continuous_error_upper_bound": maximum + BASE_ROUNDOFF_BOUND,
        "worst_vertex": VERTICES[index].tolist(), "worst_vertex_index": index,
        "vertex_count": len(VERTICES), "subset_count": len(SUBSETS),
        "vertex_values_sha256": _sha(values.tobytes()), "independent_certificate": certificate,
        "independent_reference_exact_match": True,
        "individual_neuron_maxima": individual.tolist(), "naive_sum_upper_bound": float(individual.sum()),
        "max_single_neuron": float(individual.max()), "smoke_max_error": float(smoke_errors.max()),
        "initial_probe_all_passed_cpu": bool(smoke_errors.max() <= TOLERANCE),
        "weights": weights.tolist(), "biases": biases.tolist(), "coefficients": coefficients.tolist(),
        "input_sha256": [_sha(array.tobytes()) for array in (smoke, weights, biases, coefficients)]}


PROBLEM = """Validate pruning of a rare-activation ReLU residual over a continuous input box.

The fixed parameters W (6,12), b (6,), and c (6,) are the actual float32
arrays generated in make_inputs_numpy(). They do not change between calls.
The legal input is ANY contiguous finite float32 matrix X of shape (n,12),
1 <= n <= 4096, whose entries all lie in [-1,1]. The workload is the whole
box, not just the smoke inputs returned by make_inputs_numpy().

For each row x the mathematical target, over its actual stored float32
values and the stored parameters, is
  target(x) = 0.25*x[0] + 0.5*x[1]
              + sum_{j=0..5} c[j] * max(dot(W[j],x) - b[j], 0).
The candidate omits the small residual module, retaining the linear base.
It is acceptable iff EVERY legal x has absolute output error <= 1.0 against
this mathematical target (evaluate the reference in float64 or better).
No relative-error test or independent per-neuron error threshold applies.

run(X,W,b,c) must return a finite float32 vector of shape (n,) without
modifying any input. There is no requirement on parameters from other seeds.
make_inputs() supplies the fixed parameters and a convenient smoke batch;
other legal X may be constructed to establish or refute the whole-box bound.

kernel.py includes initial_probe(), an executable prior experiment. Its
code and observed result are evidence to assess, not an authoritative verdict
or a replacement for the universal-domain requirement. All evaluation
systems receive the same code, contract, and initial observations.
"""


KERNEL_TEMPLATE = '''import numpy as np
import torch
import triton
import triton.language as tl

SEED = __SEED__


@triton.jit
def _pruned_ffn(X, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x0 = tl.load(X + row * 12, row < N, other=0.0)
    x1 = tl.load(X + row * 12 + 1, row < N, other=0.0)
    value = 0.25 * x0 + 0.5 * x1
    tl.store(Out + row, value, row < N)


def run(x, weights, biases, coefficients):
    output = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
    _pruned_ffn[(triton.cdiv(x.shape[0], 128),)](
        x, output, N=x.shape[0], BLOCK=128, enable_fp_fusion=False)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(SEED))
    center = rng.choice(np.asarray([-1, 1]), size=12)
    flip_probability = rng.uniform(0.05, 0.4)
    flips = np.where(rng.uniform(size=(6, 12)) < flip_probability, -1, 1)
    magnitudes = rng.choice(np.asarray([0.5, 1.0]), size=(6, 12))
    weights = (center[None, :] * flips * magnitudes).astype(np.float32)
    biases = (0.75 * np.abs(weights).sum(axis=1)).astype(np.float32)
    coefficients = np.full(6, 0.25, dtype=np.float32)
    smoke = np.concatenate((np.zeros((1, 12)), np.eye(12), -np.eye(12),
                            rng.uniform(-1.0, 1.0, size=(16, 12))), axis=0).astype(np.float32)
    return smoke, weights, biases, coefficients


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(value).to(device) for value in make_inputs_numpy())


def initial_probe():
    inputs = make_inputs()
    before = [value.clone() for value in inputs]
    actual = run(*inputs)
    x, w, b, c = [value.double() for value in inputs]
    expected = 0.25*x[:, 0] + 0.5*x[:, 1] + torch.relu(x @ w.T - b) @ c
    error = torch.abs(actual.double() - expected)
    structural = (actual.dtype == torch.float32 and actual.shape == (41,)
                  and bool(torch.isfinite(actual).all())
                  and all(torch.equal(a, z) for a, z in zip(inputs, before)))
    return {"scope": "41 smoke rows: origin, signed coordinate axes, and 16 uniform box samples",
            "max_absolute_error": float(error.max()), "tolerance": 1.0,
            "shape_dtype_finite_and_inputs_unmodified": bool(structural),
            "all_passed": bool(structural and float(error.max()) <= 1.0)}
'''


def _write_frozen(path, text):
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build(root):
    root = Path(root)
    candidates = [_candidate(seed) for seed in SEARCH_SEEDS]
    search = {"family": "box_relu_audit", "numpy_version": np.__version__,
        "search_seeds": list(SEARCH_SEEDS), "candidates": candidates,
        "calibration": "No preliminary calibration; fixed dyadic parameter generator, 64 seeds, tolerance 1 and 0.75/1.25 selection margins before this search.",
        "selection_rule": "First pass with M<=0.75 and first failure with M>=1.25; both require smoke error<=0.1, naive sum>1, and every individual maximum<1.",
        "llm_feedback_used": False}
    eligible = [row for row in candidates if row["smoke_max_error"] <= 0.1
                and row["naive_sum_upper_bound"] > TOLERANCE and row["max_single_neuron"] < TOLERANCE]
    passing = [row for row in eligible if row["error"] <= 0.75]
    failing = [row for row in eligible if row["error"] >= 1.25]
    if not passing or not failing:
        search["construction_failed"] = True
        _write_frozen(root / "private_data" / "search_box_relu.json", json.dumps(search, indent=2) + "\n")
        raise ValueError("Fixed recorded search did not produce required margins")
    selected = dict(zip(CASES, (passing[0], failing[0])))
    answer = {"family": "box_relu_audit", "tolerance": TOLERANCE,
              "base_roundoff_bound": BASE_ROUNDOFF_BOUND, "cases": {}}
    for name, row in selected.items():
        code = KERNEL_TEMPLATE.replace("__SEED__", str(row["seed"]))
        folder = root.parent / "triton_eval_cases" / name
        _write_frozen(folder / "kernel.py", code)
        _write_frozen(folder / "problem.txt", PROBLEM)
        _write_frozen(folder / "meta.json", json.dumps({"name": name, "passed": None,
            "status": "unverified", "family": "box_relu_audit"}, indent=2) + "\n")
        answer["cases"][name] = dict(row, kernel_sha256=_sha(code.encode()),
            problem_sha256=_sha(PROBLEM.encode()), base_roundoff_bound=BASE_ROUNDOFF_BOUND,
            mechanism="A continuous-box guarantee needs the joint convex maximum; random smoke tests miss it and summing incompatible per-neuron maxima overestimates it.")
    search["selected"] = {name: row["seed"] for name, row in selected.items()}
    _write_frozen(root / "private_data" / "answer_key_box_relu.json", json.dumps(answer, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_box_relu.json", json.dumps(search, indent=2) + "\n")
    return answer


def validate_case(name, kernel):
    import torch

    if name not in CASES:
        raise ValueError(name)
    arrays = kernel.make_inputs_numpy()
    if any(not np.array_equal(a, b) for a, b in zip(arrays, _inputs(kernel.SEED))):
        raise ValueError("Public/private input generation differs")
    canonical = [_sha(a.tobytes()) for a in arrays]
    row = _candidate(kernel.SEED)
    _, weights, biases, coefficients = arrays
    target = base(VERTICES) + residual(VERTICES, weights, biases, coefficients)
    repeats = []
    for repeat in range(10):
        inputs = list(kernel.make_inputs())
        if [_sha(a.detach().cpu().numpy().tobytes()) for a in inputs] != canonical:
            raise ValueError("Actual GPU canonical input bytes differ")
        inputs[0] = torch.from_numpy(VERTICES.copy()).to("cuda")
        hashes = [_sha(a.detach().cpu().numpy().tobytes()) for a in inputs]
        result = kernel.run(*inputs)
        actual = result.detach().cpu().numpy()
        if result.dtype != torch.float32 or tuple(result.shape) != (len(VERTICES),) or not np.isfinite(actual).all():
            raise ValueError("GPU shape/dtype/finite violation")
        if not np.array_equal(actual, emulate(VERTICES)):
            raise ValueError("Actual GPU base differs from exact vertex emulation")
        if [_sha(a.detach().cpu().numpy().tobytes()) for a in inputs] != hashes:
            raise ValueError("GPU inputs changed")
        error = float(np.max(np.abs(actual.astype(np.float64) - target)))
        if error != row["error"]:
            raise ValueError("Actual GPU vertex maximum differs from independent certificates")
        repeats.append({"repeat": repeat, "error": error, "output_sha256": _sha(actual.tobytes())})
    torch.cuda.synchronize()
    if len({r["output_sha256"] for r in repeats}) != 1:
        raise ValueError("GPU repeats differ")
    probe = kernel.initial_probe()
    if not probe["all_passed"]:
        raise ValueError("Initial smoke unexpectedly fails")
    return {"case": name, "ground_truth": row["cpu_ground_truth"], "seed": kernel.SEED,
        "input_sha256": canonical, "gpu_domain_input_sha256": _sha(VERTICES.tobytes()),
        "error": row["error"], "budget": TOLERANCE, "tolerance": TOLERANCE,
        "continuous_domain_certificate": {
            "method": "Convex residual max over 4096 box vertices equals exact integer max over 64 active-subset support functions.",
            "residual_maximum": row["error"], "candidate_base_roundoff_bound": BASE_ROUNDOFF_BOUND,
            "whole_box_error_upper_bound": row["continuous_error_upper_bound"],
            "roundoff_reason": "Finite float32 |x|<=1 only. The two binary-power scalings are exact outside subnormals, with combined possible underflow/FTZ error below 2^-125. Final |base|<=0.75 addition has rounding error at most 2^-25 (plus at most 2^-126 for possible sum FTZ), so total <1e-7. Vertex base arithmetic is exact.",
            "independent_integer_certificate": row["independent_certificate"]},
        "worst_input": row["worst_vertex"], "vertex_count": len(VERTICES), "active_subset_count": len(SUBSETS),
        "naive_sum_upper_bound": row["naive_sum_upper_bound"],
        "individual_neuron_maxima": row["individual_neuron_maxima"],
        "independent_reference_exact_match": True, "output_contract_shape_dtype_finite": True,
        "gpu_inputs_unchanged": True, "repeat_count": 10, "repeatable": True,
        "repeats": repeats, "output_sha256": repeats[0]["output_sha256"], "initial_probe": probe}


if __name__ == "__main__":
    answers = build(Path(__file__).resolve().parents[3] / "solo_vs_debate_challenges")
    print(json.dumps({name: {key: row[key] for key in (
        "seed", "cpu_ground_truth", "error", "smoke_max_error", "naive_sum_upper_bound", "max_single_neuron")}
        for name, row in answers["cases"].items()}, indent=2))
