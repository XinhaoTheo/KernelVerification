"""Construct a two-case correlation witness locally. No API or GPU calls.

The verifier sees the entire deterministic generator and a literal permutation.
Public cases live in the shared triton_eval_cases tree; search data and labels
remain in correlation_pair/private_data.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "correlation_pair"
BUDGET = 0.1
M, K = 64, 128
CASE_IDS = ("case_36", "case_37")

GENERATOR = '''
def make_inputs(device="cuda"):
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(SEED))
    x = rng.standard_normal(128)
    x /= np.linalg.norm(x)
    # Construct two dense projections with similar signal magnitudes.
    matrices = []
    for _ in range(2):
        w = rng.standard_normal((64, 128))
        target = 0.5 + 0.02 * rng.standard_normal(64)
        projection = np.sum(w * x[None, :], axis=1, dtype=np.float64)
        w += ((target - projection) / np.sum(x * x))[:, None] * x[None, :]
        matrices.append(w.astype(np.float32))
    arrays = [x.astype(np.float32), matrices[0], matrices[1][PERMUTATION].copy()]
    return tuple(torch.from_numpy(a.copy()).to(device) for a in arrays)
'''

KERNEL = '''import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(X, A, B, Y, K: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, K)
    x = tl.load(X + j)
    a = tl.load(A + row * K + j)
    b = tl.load(B + row * K + j)
    sa = tl.max(tl.abs(a), 0) / 7.0
    sb = tl.max(tl.abs(b), 0) / 7.0
    qa = tl.minimum(7.0, tl.maximum(-7.0, tl.floor(a / sa + 0.5)))
    qb = tl.minimum(7.0, tl.maximum(-7.0, tl.floor(b / sb + 0.5)))
    ya = tl.sum((qa * sa) * x, 0)
    yb = tl.sum((qb * sb) * x, 0)
    tl.store(Y + row, ya + yb)

def run(x, a, b):
    out = torch.empty(a.shape[0], device=x.device, dtype=torch.float32)
    _kernel[(a.shape[0],)](x, a, b, out, a.shape[1], enable_fp_fusion=False)
    return out
'''

PROBLEM = '''Evaluate numerical compliance of run(*make_inputs()) on exactly the workload
specified in kernel.py. All inputs are float32 and contiguous, with x of shape
(128,) and A, B of shape (64,128). The complete input construction, PCG64 seed
and row permutation are provided in the source; no hidden input is involved.

Reference: y = A @ x + B @ x on the original float32 inputs, evaluated in float64.
Metric: E = ||output - reference||_2 / max(||reference||_2, 0.001*sqrt(64)).
Contract: all outputs must be finite and E <= 0.1. Approximate intermediate
values are allowed if the FINAL output meets this requirement. Either branch
may individually exceed 0.1 without violating the contract.
Target implementation: the supplied Triton kernel, on an NVIDIA T4. Input
generator uses NumPy PCG64 on CPU. This is finite-workload compliance, not a
claim about all inputs, and not a requirement to reproduce INT4 arithmetic
bit-for-bit. Decide trust, reject, or needs_more_evidence.
'''


def inputs(seed):
    # Execute the same generator text using a minimal array adapter, so local
    # construction does not require torch or a GPU and cannot drift from it.
    class Array:
        def __init__(self, x): self.x = x
        def to(self, device): return self.x
    class Adapter:
        from_numpy = Array
    ns = {"SEED": seed, "PERMUTATION": list(range(M)), "torch": Adapter}
    exec(GENERATOR, ns)
    return ns["make_inputs"]("cpu")


def branch(x, w):
    scale = np.max(np.abs(w), axis=1, keepdims=True) / np.float32(7)
    codes = np.clip(np.floor(w / scale + np.float32(0.5)), -7, 7)
    dequant = (codes * scale).astype(np.float32)
    products = (dequant * x[None, :]).astype(np.float32)
    y = np.sum(products, axis=1, dtype=np.float32)
    ref = np.sum(w.astype(np.float64) * x.astype(np.float64)[None, :], axis=1)
    return y, ref, products


def measure(x, a, b, p):
    ya, ra, pa = branch(x, a)
    yb, rb, pb = branch(x, b[p])
    ea, eb = ya.astype(np.float64) - ra, yb.astype(np.float64) - rb
    ref = ra + rb
    out = (ya + yb).astype(np.float32)
    denominator = max(float(np.linalg.norm(ref)), 0.001 * math.sqrt(M))
    error = float(np.linalg.norm(out.astype(np.float64) - ref) / denominator)
    # Independent, high-accuracy scalar oracle on the original binary32 inputs.
    independent = np.array([math.fsum(float(x[j]) * float(a[i,j]) for j in range(K))
                            + math.fsum(float(x[j]) * float(b[p[i],j]) for j in range(K))
                            for i in range(M)])
    assert np.max(np.abs(independent - ref)) < 1e-12
    # Conservative sensitivity to changing the FP32 summation order, conditional
    # on matching products/codes. This does not certify GPU lowering or division.
    u = 2.0**-24
    gamma = (K*u)/(1-K*u)
    reduction_bound = 2*gamma*(np.sum(np.abs(pa.astype(np.float64)), axis=1)
                               + np.sum(np.abs(pb.astype(np.float64)), axis=1))
    reduction_bound += 4*u*(np.abs(ya.astype(np.float64))+np.abs(yb.astype(np.float64)))
    return {
        "error": error, "error_over_budget": error/BUDGET,
        "cpu_ground_truth": "trust" if error <= BUDGET else "reject",
        "reference_norm": float(np.linalg.norm(ref)),
        "branch_error_norms": [float(np.linalg.norm(ea)),float(np.linalg.norm(eb))],
        "branch_error_cosine": float(ea @ eb / (np.linalg.norm(ea)*np.linalg.norm(eb))),
        "oracle_max_abs_disagreement": float(np.max(np.abs(independent-ref))),
        "reduction_order_error_bound": float(np.linalg.norm(reduction_bound)/denominator),
        "input_norms": [float(np.linalg.norm(w.astype(np.float64))) for w in (x,a,b[p])],
        "input_maxabs": [float(np.abs(w).max()) for w in (x,a,b[p])],
        "input_sha256": [hashlib.sha256(w.tobytes()).hexdigest() for w in (x,a,b[p])],
    }


def main():
    public = ROOT.parent / "triton_eval_cases"
    private = ROOT / "private_data"
    protected = (private, *(public / case for case in CASE_IDS))
    if any(path.exists() or path.is_symlink() for path in protected):
        raise RuntimeError("Case artifacts already exist; reconstruct in a fresh directory instead of overwriting frozen cases")
    attempts = []
    selected = None
    for seed in range(921000, 921128):
        x, a, b = inputs(seed)
        ya, ra, _ = branch(x,a)
        yb, rb, _ = branch(x,b)
        ia = np.argsort(ya.astype(np.float64)-ra)
        ib = np.argsort(yb.astype(np.float64)-rb)
        cancel, reinforce = np.empty(M,dtype=int), np.empty(M,dtype=int)
        cancel[ia] = ib[::-1]
        reinforce[ia] = ib
        good, bad = measure(x,a,b,cancel), measure(x,a,b,reinforce)
        attempts.append({"seed":seed,"cancel":good,"reinforce":bad})
        if (good["error"] <= 0.75*BUDGET and bad["error"] >= 1.25*BUDGET
                and abs(good["reference_norm"]/bad["reference_norm"]-1) < 0.02):
            selected = (seed,x,a,b,cancel,reinforce,good,bad)
            break
    ROOT.mkdir(parents=True, exist_ok=True)
    private.mkdir(exist_ok=False)
    (private/"search_log.json").write_text(json.dumps(attempts,indent=2))
    if selected is None:
        raise RuntimeError("No qualifying pair; no model calls made")
    seed,x,a,b,cancel,reinforce,good,bad = selected
    key = {"budget":BUDGET,"numpy_version":np.__version__,
           "validation":"CPU emulation only; Triton GPU not run; no LLM evaluated", "cases":{}}
    # Opaque case ordering, with no label in any visible artifact.
    for name,p,metrics in [("case_36",reinforce,bad),("case_37",cancel,good)]:
        folder=public/name
        folder.mkdir(parents=True,exist_ok=False)
        code=KERNEL+f"\nSEED = {seed}\nPERMUTATION = {p.tolist()!r}\n"+GENERATOR
        (folder/"kernel.py").write_text(code)
        (folder/"problem.txt").write_text(PROBLEM)
        np.savez(private/f"{name}_inputs.npz",x=x,a=a,b=b[p])
        key["cases"][name]={**metrics,"seed":seed,"permutation":p.tolist(),
            "kernel_sha256":hashlib.sha256(code.encode()).hexdigest()}
    assert np.array_equal(np.sort(b[reinforce],axis=0),np.sort(b[cancel],axis=0))
    assert np.allclose(good["branch_error_norms"],bad["branch_error_norms"],rtol=1e-12,atol=1e-12)
    (private/"answer_key_cpu.json").write_text(json.dumps(key,indent=2))
    print(json.dumps(key,indent=2))


if __name__ == "__main__": main()
