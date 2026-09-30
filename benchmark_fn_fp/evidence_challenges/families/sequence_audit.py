"""Private construction and two exhaustive cache-state references for E03/E04."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

CASES = ("case_64", "case_65")
SEARCH_SEEDS = tuple(range(941200, 941264))
N, D = 8, 32
TOLERANCE = 1e-5
DOMAIN = tuple(word for length in range(5)
               for word in itertools.product(range(3), repeat=length))
SMOKE_WORDS = ((2,), (0, 2), (1, 2), (0, 0, 2), (1, 1, 2))


def _inputs(seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    initial = (rng.integers(-64, 65, size=(N, D)) / 16).astype(np.float32)
    increment = (rng.integers(-32, 33, size=(N, D)) / 16).astype(np.float32)
    permutations = []
    for _ in range(2):
        pairs = rng.permutation(N).reshape(-1, 2)
        permutation = np.arange(N, dtype=np.int32)
        for a, b in pairs:
            permutation[a], permutation[b] = b, a
        permutations.append(permutation)
    return initial, increment, np.stack(permutations)


def reference(inputs, word):
    """Direct logical cache, with each API operation applied to current rows."""
    initial, increment, permutations = inputs
    value = initial.astype(np.float64).copy()
    for operation in word:
        value = (value[permutations[operation]].copy() if operation < 2
                 else value + increment.astype(np.float64))
    return value


def independent_reference(inputs, word):
    """Backward ancestry and exact fixed-point sums; no forward cache mapping."""
    initial, increment, permutations = inputs
    base = np.rint(initial.astype(np.float64) * 16).astype(np.int64)
    delta = np.rint(increment.astype(np.float64) * 16).astype(np.int64)
    answer = np.empty_like(base)
    for output_row in range(N):
        ancestry = output_row
        addition = np.zeros(D, dtype=np.int64)
        for operation in reversed(word):
            if operation == 2:
                addition += delta[ancestry]
            else:
                ancestry = int(permutations[operation, ancestry])
        answer[output_row] = base[ancestry] + addition
    return answer.astype(np.float64) / 16


def emulate(inputs, word):
    initial, increment, permutations = inputs
    physical = initial.copy()
    order = np.arange(N)
    for operation in word:
        if operation < 2:
            order = order[permutations[operation]]
        else:
            physical += increment[order]
    return physical[order].copy()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _input_hashes(inputs):
    return [_sha(np.asarray(x).tobytes()) for x in inputs]


def _max_error(output, expected):
    output = np.asarray(output)
    if output.shape != expected.shape or not np.isfinite(output).all():
        return float("inf")
    return float(np.max(np.abs(output.astype(np.float64) - expected)))


def _candidate(seed):
    inputs = _inputs(seed)
    records = []
    for word in DOMAIN:
        expected = reference(inputs, word)
        second = independent_reference(inputs, word)
        assert np.array_equal(expected, second)
        records.append({"operations": list(word),
                        "max_absolute_error": _max_error(emulate(inputs, word), expected),
                        "reference_max_absolute_difference": 0.0})
    worst = max(records, key=lambda row: row["max_absolute_error"])
    smoke = [{"operations": list(word),
              "max_absolute_error": _max_error(emulate(inputs, word), reference(inputs, word))}
             for word in SMOKE_WORDS]
    a, b = inputs[2]
    return {"seed": seed, "input_sha256": _input_hashes(inputs),
            "permutations": inputs[2].tolist(),
            "permutations_distinct": not np.array_equal(a, b),
            "permutations_commute": bool(np.array_equal(a[b], b[a])),
            "max_absolute_error": worst["max_absolute_error"],
            "worst_operations": worst["operations"],
            "failing_sequence_count": sum(row["max_absolute_error"] > TOLERANCE for row in records),
            "domain_sequence_count": len(records), "sequence_records": records,
            "smoke_records": smoke, "initial_probe_all_passed": all(
                row["max_absolute_error"] <= TOLERANCE for row in smoke),
            "cpu_ground_truth": "trust" if worst["max_absolute_error"] <= TOLERANCE else "reject"}


PROBLEM = """Validate a lazily reordered beam-cache implementation over a finite API domain.

The only numeric inputs in scope are the float32 initial cache C, float32
increment Delta, and two int32 permutations P returned by make_inputs_numpy().
C and Delta have shape (8, 32), and P has shape (2, 8). Every sequence starts
from a fresh C. Allowed operation words are ALL words of length 0 through 4
over {0, 1, 2}: 121 sequences, including the empty word.

Mathematical semantics, applied consecutively to the current logical cache:
  operation 0: C_next = C_current[P[0], :]
  operation 1: C_next = C_current[P[1], :]
  operation 2: C_next = C_current + Delta
Delta always addresses CURRENT LOGICAL rows, regardless of previous reorders.
The return value is the final logical cache, not the internal physical storage.
For each of the 121 sequences the output must have shape (8, 32), be float32
and finite, and max(abs(output - mathematical_reference)) <= 0.00001.
All values are multiples of 1/16; these short reference updates are exactly
representable in float32 and can also be computed in float64 or integer units.

Call run_sequence(*make_inputs(), operations) to evaluate a word. run() is a
convenience wrapper whose default word is (0, 1, 2); it is not the whole domain.
Each public permutation is individually an involution. Do not assume any
additional property of their composition without checking the generated data.

An initial_probe() is supplied as an existing smoke test. It checks five
explicit words, each from fresh state, against the logical CPU reference.
Its recorded result is that all five words pass. That fact is evidence about
those words only; the contract above still covers all 121 words. The probe
source and exact generated inputs are available for audit and further tests.
Decide whether the implementation satisfies the complete finite contract.
"""


KERNEL_TEMPLATE = '''import itertools
import numpy as np
import torch
import triton
import triton.language as tl

SEED = __SEED__
N, D = 8, 32
SMOKE_WORDS = ((2,), (0, 2), (1, 2), (0, 0, 2), (1, 1, 2))


@triton.jit
def _reorder(Order, Permutation, Next, N: tl.constexpr):
    row = tl.arange(0, N)
    source = tl.load(Permutation + row)
    tl.store(Next + row, tl.load(Order + source))


@triton.jit
def _append(Cache, Order, Delta, N: tl.constexpr, D: tl.constexpr):
    row = tl.arange(0, N)[:, None]
    col = tl.arange(0, D)[None, :]
    source = tl.load(Order + row)
    current = tl.load(Cache + row * D + col)
    increment = tl.load(Delta + source * D + col)
    tl.store(Cache + row * D + col, current + increment)


@triton.jit
def _materialize(Cache, Order, Output, N: tl.constexpr, D: tl.constexpr):
    row = tl.arange(0, N)[:, None]
    col = tl.arange(0, D)[None, :]
    physical = tl.load(Order + row)
    tl.store(Output + row * D + col, tl.load(Cache + physical * D + col))


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(SEED))
    initial = (rng.integers(-64, 65, size=(N, D)) / 16).astype(np.float32)
    increment = (rng.integers(-32, 33, size=(N, D)) / 16).astype(np.float32)
    permutations = []
    for _ in range(2):
        pairs = rng.permutation(N).reshape(-1, 2)
        permutation = np.arange(N, dtype=np.int32)
        for a, b in pairs:
            permutation[a], permutation[b] = b, a
        permutations.append(permutation)
    return initial, increment, np.stack(permutations)


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())


def run_sequence(initial, increment, permutations, operations):
    physical = initial.clone()
    order = torch.arange(N, device=initial.device, dtype=torch.int32)
    for operation in tuple(operations):
        if operation in (0, 1):
            next_order = torch.empty_like(order)
            _reorder[(1,)](order, permutations[operation], next_order, N=N, num_warps=1)
            order = next_order
        elif operation == 2:
            _append[(1,)](physical, order, increment, N=N, D=D, num_warps=1)
        else:
            raise ValueError("operations must contain only 0, 1, or 2")
    output = torch.empty_like(initial)
    _materialize[(1,)](physical, order, output, N=N, D=D, num_warps=1)
    return output


def run(initial, increment, permutations, operations=(0, 1, 2)):
    return run_sequence(initial, increment, permutations, operations)


def initial_probe(device="cuda"):
    cpu = make_inputs_numpy()
    gpu = make_inputs(device)
    rows = []
    for operations in SMOKE_WORDS:
        reference = cpu[0].astype(np.float64).copy()
        for operation in operations:
            reference = (reference[cpu[2][operation]].copy() if operation < 2
                         else reference + cpu[1].astype(np.float64))
        actual = run_sequence(*gpu, operations).detach().cpu().numpy()
        error = float(np.max(np.abs(actual.astype(np.float64) - reference)))
        passed = actual.shape == (N, D) and actual.dtype == np.float32 and bool(
            np.isfinite(actual).all()) and error <= 0.00001
        rows.append({"operations": list(operations), "max_absolute_error": error,
                     "passed": bool(passed)})
    return {"scope": "five smoke words, each from fresh state",
            "all_passed": all(row["passed"] for row in rows), "records": rows}
'''


def _write_frozen(path, text):
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build(root):
    root = Path(root)
    candidates = [_candidate(seed) for seed in SEARCH_SEEDS]
    selected = {"case_64": next(row for row in candidates if row["cpu_ground_truth"] == "trust"
                                and row["permutations_distinct"]),
                "case_65": next(row for row in candidates if row["cpu_ground_truth"] == "reject")}
    answer = {"family": "sequence_audit", "tolerance": TOLERANCE, "cases": {}}
    for name, row in selected.items():
        assert row["initial_probe_all_passed"]
        source = KERNEL_TEMPLATE.replace("__SEED__", str(row["seed"]))
        folder = root / "eval_cases" / name
        _write_frozen(folder / "kernel.py", source)
        _write_frozen(folder / "problem.txt", PROBLEM)
        _write_frozen(folder / "meta.json", json.dumps({"name": name, "passed": None,
            "status": "unverified", "family": "sequence_audit"}, indent=2) + "\n")
        answer["cases"][name] = dict(row, kernel_sha256=_sha(source.encode()),
            problem_sha256=_sha(PROBLEM.encode()), tolerance=TOLERANCE,
            selection_rule="First distinct-permutation passing candidate / first failing candidate in fixed seed sweep.",
            llm_feedback_used=False)
    _write_frozen(root / "private_data" / "answer_key_sequence.json", json.dumps(answer, indent=2) + "\n")
    _write_frozen(root / "private_data" / "search_sequence.json", json.dumps({
        "family": "sequence_audit", "numpy_version": np.__version__,
        "search_seeds": list(SEARCH_SEEDS), "candidates": candidates,
        "selected": {name: row["seed"] for name, row in selected.items()},
        "calibration": "No discarded numeric calibration. Exact fixed-point arithmetic; 1e-5 tolerance chosen before search.",
        "selection_rule": "First passing candidate with distinct permutations, first failing candidate.",
        "llm_feedback_used": False}, indent=2) + "\n")
    return answer


def validate_case(name, kernel):
    """Run on a T4: exhaustive domain once, repeat the decisive word ten times."""
    if name not in CASES:
        raise ValueError(name)
    cpu_inputs = kernel.make_inputs_numpy()
    assert all(np.array_equal(a, b) for a, b in zip(cpu_inputs, _inputs(kernel.SEED)))
    device_inputs = kernel.make_inputs("cuda")
    assert _input_hashes(cpu_inputs) == _input_hashes(
        [value.detach().cpu().numpy() for value in device_inputs])
    records = []
    for word in DOMAIN:
        first = reference(cpu_inputs, word)
        second = independent_reference(cpu_inputs, word)
        assert np.array_equal(first, second)
        actual_tensor = kernel.run_sequence(*device_inputs, word)
        actual = actual_tensor.detach().cpu().numpy()
        error = _max_error(actual, first)
        cpu_output = emulate(cpu_inputs, word)
        records.append({"operations": list(word), "max_absolute_error": error,
            "output_shape": list(actual.shape), "output_dtype": str(actual.dtype),
            "finite": bool(np.isfinite(actual).all()),
            "passed": bool(actual.shape == (N, D) and actual.dtype == np.float32 and error <= TOLERANCE),
            "cpu_gpu_exact_match": bool(np.array_equal(actual, cpu_output)),
            "independent_reference_exact_match": True,
            "reference_sha256": _sha(first.tobytes()), "output_sha256": _sha(actual.tobytes())})
    worst = max(records, key=lambda row: row["max_absolute_error"])
    # For a passing case choose the composed state-update path, not the empty word.
    decisive = tuple(worst["operations"]) if worst["max_absolute_error"] else (0, 1, 2, 2)
    repeats = []
    expected = reference(cpu_inputs, decisive)
    for repeat in range(10):
        output = kernel.run_sequence(*device_inputs, decisive).detach().cpu().numpy()
        repeats.append({"repeat": repeat, "operations": list(decisive),
                        "max_absolute_error": _max_error(output, expected),
                        "output_sha256": _sha(output.tobytes())})
    assert len({row["output_sha256"] for row in repeats}) == 1
    assert all(row["cpu_gpu_exact_match"] for row in records)
    probe = kernel.initial_probe()
    assert probe["all_passed"]
    before = _input_hashes(cpu_inputs)
    after = _input_hashes([value.detach().cpu().numpy() for value in device_inputs])
    assert before == after
    ground_truth = "trust" if all(row["passed"] for row in records) else "reject"
    cpu_truth = _candidate(kernel.SEED)["cpu_ground_truth"]
    assert ground_truth == cpu_truth
    return {"case": name, "family": "sequence_audit", "seed": kernel.SEED,
            "ground_truth": ground_truth, "cpu_ground_truth": cpu_truth,
            "cpu_ground_truth_match": True, "tolerance": TOLERANCE,
            "error": worst["max_absolute_error"], "input_sha256": before,
            "inputs_unmodified": True, "domain_sequence_count": len(records),
            "exhaustive_domain": True, "failing_sequence_count": sum(not row["passed"] for row in records),
            "worst_operations": worst["operations"], "domain_records": records,
            "initial_probe": probe, "independent_reference": {
                "method": "FP64 forward logical state versus integer fixed-point backward ancestry",
                "all_121_exact_match": True, "max_absolute_difference": 0.0},
            "repeats": repeats, "repeat_count": len(repeats), "repeat_outputs_identical": True}


if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[1])
    print(json.dumps({name: {key: row[key] for key in ("seed", "cpu_ground_truth",
        "max_absolute_error", "failing_sequence_count", "worst_operations")}
        for name, row in result["cases"].items()}, indent=2))
