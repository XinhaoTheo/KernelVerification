"""Finite-workload summation order witnesses; construction side only."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

CASES = ("case_42", "case_43")
TOLERANCE = 0.1
SEED = 730119

GENERATOR = '''
def make_inputs_numpy():
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(SEED))
    w = rng.integers(1, 33, size=(64, 32)).astype(np.float32) * np.float32(1048576)
    small = rng.integers(1, 4, size=(64, 64)).astype(np.float32) * np.float32(0.25)
    values = np.concatenate([w, -w, small], axis=1)
    order = rng.permutation(128)
    values = values[:, order][:, PERMUTATION].copy()
    return (values,)

def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(a.copy()).to(device) for a in make_inputs_numpy())
'''

KERNEL = '''import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(X, Y, K: tl.constexpr, R: tl.constexpr):
    rows = tl.arange(0, R)
    acc = tl.full((R,), 0, tl.float32)
    for j in range(K):
        value = tl.load(X + rows * K + j)
        acc = acc + value
    tl.store(Y + rows, acc)

def run(values):
    out = torch.empty((values.shape[0],), dtype=torch.float32, device=values.device)
    _kernel[(1,)](values, out, values.shape[1], values.shape[0], enable_fp_fusion=False)
    return out
'''

PROBLEM = '''Evaluate numerical compliance of run(*make_inputs()) on exactly the
fully specified workload in kernel.py. There is one contiguous float32 input
matrix of shape (64,128). The PCG64 seed and complete column permutation are
public; no hidden inputs are used.

Reference: for each row, sum the ORIGINAL float32 entries in real arithmetic,
evaluated with float64 or an independently checked high-accuracy summation.
Metric: E = ||output-reference||_2 / max(||reference||_2, 0.008).
Contract: every output must be finite and E <= 0.1. Only the final result on
the specified input is in scope. The kernel may use float32 arithmetic and any
summation order if this numerical requirement is met. This is not a universal
correctness claim about all possible inputs.
Target: the supplied Triton kernel on an NVIDIA T4. Decide trust, reject, or
needs_more_evidence.
'''


def inputs(permutation):
    ns = {"SEED": SEED, "PERMUTATION": list(permutation)}
    exec(GENERATOR, ns)
    return ns["make_inputs_numpy"]()


def reference(arrays):
    return np.sum(np.asarray(arrays[0], dtype=np.float64), axis=1, dtype=np.float64)


def independent_reference(arrays):
    return np.array([math.fsum(map(float, row)) for row in arrays[0]], dtype=np.float64)


def emulate(arrays):
    x = arrays[0]
    out = np.zeros(x.shape[0], dtype=np.float32)
    for j in range(x.shape[1]):
        out = (out + x[:, j]).astype(np.float32)
    return out


def metric(output, ref):
    return float(np.linalg.norm(np.asarray(output, dtype=np.float64) - ref) /
                 max(float(np.linalg.norm(ref)), 0.008))


def build(output_root):
    root = Path(output_root)
    for name in CASES:
        if (root / "eval_cases" / name).exists():
            raise FileExistsError(f"Do not overwrite frozen case: {name}")
    rng = np.random.Generator(np.random.PCG64(SEED))
    rng.integers(1, 33, size=(64, 32))
    rng.integers(1, 4, size=(64, 64))
    shuffled = rng.permutation(128)
    inverse = np.argsort(shuffled)
    # Reconstruct a low-error order only on the construction side. The agent
    # receives its literal indices, never this selection recipe or its label.
    pairing = np.random.Generator(np.random.PCG64(SEED + 1)).permutation(32)
    ordered = np.array([j for i in pairing for j in (i, i + 32)] + list(range(64, 128)))
    selected = {"case_42": np.random.Generator(np.random.PCG64(SEED + 2)).permutation(128),
                "case_43": inverse[ordered]}
    key = {"family": "reduction", "budget": TOLERANCE, "cases": {}}
    attempts = []
    for name, permutation in selected.items():
        arrays = inputs(permutation)
        ref = reference(arrays)
        independent = independent_reference(arrays)
        assert np.array_equal(ref, independent)
        output = emulate(arrays)
        error = metric(output, ref)
        label = "trust" if error <= TOLERANCE else "reject"
        assert error <= 0.75*TOLERANCE or error >= 1.25*TOLERANCE
        code = KERNEL + f"\nSEED = {SEED}\nPERMUTATION = {permutation.tolist()!r}\n" + GENERATOR
        dest = root / "eval_cases" / name
        dest.mkdir(parents=True)
        (dest / "kernel.py").write_text(code)
        (dest / "problem.txt").write_text(PROBLEM)
        (dest / "meta.json").write_text(json.dumps({"name": name, "status": "under_test", "passed": None}, indent=2))
        measurement = {"cpu_ground_truth": label, "error": error, "budget": TOLERANCE,
            "kernel_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "problem_sha256": hashlib.sha256(PROBLEM.encode()).hexdigest(),
            "input_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
            "oracle_max_abs_disagreement": float(np.max(np.abs(ref - independent))),
            "reference_norm": float(np.linalg.norm(ref)), "permutation": permutation.tolist()}
        key["cases"][name] = measurement
        attempts.append({"case": name, **measurement})
    assert {row["cpu_ground_truth"] for row in key["cases"].values()} == {"trust", "reject"}
    (root / "private_data").mkdir(parents=True, exist_ok=True)
    (root / "private_data" / "answer_key_reduction.json").write_text(json.dumps(key, indent=2))
    (root / "private_data" / "search_log_reduction.json").write_text(json.dumps(attempts, indent=2))
    return key


if __name__ == "__main__":
    result = build(Path(__file__).resolve().parents[1])
    print(json.dumps({case: {k: row[k] for k in ("cpu_ground_truth", "error", "budget")}
                      for case, row in result["cases"].items()}, indent=2))
