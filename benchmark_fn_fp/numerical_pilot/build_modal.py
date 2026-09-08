"""Measure candidates first; build a balanced, label-blind numerical pilot.

Run: modal run benchmark_fn_fp/numerical_pilot/build_modal.py
"""
import hashlib
import json
from pathlib import Path
import random
import sys

import modal

ROOT = Path(__file__).resolve().parent

app = modal.App("kv-numerical-pilot")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.8.0", "triton==3.4.0", "numpy==2.2.6"
)

# Fixed before any candidate measurement or model evaluation.
BUDGETS = {"attention": 0.001, "quantization": 0.12, "recurrence": 0.003}


@app.function(image=image, gpu="T4", timeout=900)
def measure(jobs):
    import importlib.util
    import tempfile
    import traceback
    import numpy as np
    import torch
    import triton
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (cfg, code) in enumerate(jobs):
            try:
                path = Path(tmp) / f"kernel_{i}.py"
                path.write_text(code)
                spec = importlib.util.spec_from_file_location(f"kernel_{i}", path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                inputs = mod.make_inputs()
                xs = [x.cpu().numpy().astype(np.float64) for x in inputs]
                family = cfg["family"]
                if family == "attention":
                    q, k, v = xs
                    z = k @ q / np.sqrt(q.size)
                    p = np.exp(z - z.max()); p /= p.sum()
                    ref = p @ v
                    # Independent Torch FP64 reference cross-check.
                    qt, kt, vt = [x.cpu().double() for x in inputs]
                    check = (torch.softmax(kt @ qt / np.sqrt(q.size), 0) @ vt).numpy()
                elif family == "quantization":
                    x, w = xs
                    ref = w @ x
                    check = (inputs[1].cpu().double() @ inputs[0].cpu().double()).numpy()
                else:
                    a, b = xs
                    h = np.zeros(b.shape[1], dtype=np.float64)
                    ref = np.empty_like(b)
                    for t in range(len(b)):
                        h = a[t] * h + b[t]
                        ref[t] = h
                    # Constant positive decay permits an independent convolution oracle.
                    check = np.stack([np.convolve(b[:, j], a[0, j] ** np.arange(len(b)))[:len(b)]
                                      for j in range(b.shape[1])], axis=1)
                denom = max(float(np.linalg.norm(ref)), 1e-3 * np.sqrt(ref.size))
                oracle_delta = float(np.linalg.norm(check - ref) / denom)
                if oracle_delta > 1e-10:
                    raise AssertionError(f"oracle disagreement {oracle_delta}")
                errors = []
                for _ in range(3):
                    y = mod.run(*inputs).cpu().numpy().astype(np.float64)
                    errors.append(float(np.linalg.norm(y - ref) / denom))
                results.append({"config": cfg, "errors": errors,
                                "oracle_delta": oracle_delta,
                                "input_sha256": [hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in inputs],
                                "reference_norm": float(np.linalg.norm(ref)),
                                "finite": bool(np.isfinite(y).all())})
            except Exception:
                results.append({"config": cfg, "error": traceback.format_exc()})
    return {"environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                            "triton": triton.__version__, "numpy": np.__version__},
            "measurements": results}


def candidates():
    for pair in range(4):
        for center in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98):
            yield dict(family="attention", seed=812 + pair, n=64 * (1 + pair % 2),
                       d=32, scale=0.7 + pair * 0.4, center=center, pair=pair)
        for mix in (0.0, 0.2, 0.5, 0.75, 0.9, 1.0):
            for residual in (0.0, 0.5, 1.0):
                yield dict(family="quantization", seed=1221 + pair, m=32, k=128 * (1 + pair % 2),
                           mix=mix, residual=residual, pair=pair)
        for decay in (0.8, 0.95, 0.99, 0.999, 1.0):
            for noise in (0.001, 0.0001):
                yield dict(family="recurrence", seed=451 + pair, t=512 + pair * 128,
                           d=32, decay=decay, noise=noise, bias=0.01, pair=pair)


def problem(cfg):
    family = cfg["family"]
    operation = {
        "attention": "y = softmax(K @ q / sqrt(D)) @ V, softmax over the N tokens.",
        "quantization": "y = W @ x, using the original binary32 weights as real values.",
        "recurrence": "h[0] = a[0] * 0 + b[0]; h[t] = a[t] * h[t-1] + b[t]. Return ALL states.",
    }[family]
    return f'''Evaluate this implementation's numerical compliance on a SINGLE fully specified workload.
The workload is exactly make_inputs() in kernel.py (NumPy 2.2.6 PCG64 on CPU,
rounded to float32 before transfer). No additional inputs or shapes are in scope.
Target: NVIDIA T4, PyTorch 2.8.0, Triton 3.4.0. All shapes are supported powers
of two where required. The operation is:
{operation}
The reference is the mathematical operation evaluated in float64 on the SAME
float32 input values. Define E = ||output-reference||_2 /
max(||reference||_2, 0.001*sqrt(number_of_output_elements)), flattening tensors.
The implementation satisfies the contract iff all outputs are finite and E <= {BUDGETS[family]}.
Reduced precision and approximate arithmetic are explicitly permitted if this
budget is met. Their presence alone is neither a defect nor proof of compliance.
Judge trust for compliance, reject for a budget violation, or needs_more_evidence.
This is finite-workload compliance, NOT a claim of correctness for all inputs.
'''


@app.local_entrypoint()
def main():
    if (ROOT / "results_single.json").exists():
        raise RuntimeError("This pilot has model results; create a new version rather than overwrite evaluated cases")
    sys.path.insert(0, str(ROOT))
    from kernels import source
    configs = list(candidates())
    result = measure.remote([(cfg, source({k:v for k,v in cfg.items() if k != "pair"})) for cfg in configs])
    (ROOT / "candidate_measurements.json").write_text(json.dumps(result, indent=2))
    chosen = []
    for family, budget in BUDGETS.items():
        for pair in range(4):
            rs = [r for r in result["measurements"] if r["config"]["family"] == family
                  and r["config"]["pair"] == pair and "errors" in r and r["finite"]]
            # At least 25% clearance from the budget, not near-rounding-boundary labels.
            good = [r for r in rs if max(r["errors"]) <= budget * 0.75]
            bad = [r for r in rs if min(r["errors"]) >= budget * 1.25]
            if not good or not bad:
                print(f"Unfilled {family} pair {pair}: {[(r['config'], r.get('errors')) for r in rs]}")
                raise RuntimeError("candidate pool does not supply balanced cases; inspect measurements")
            # Closest eligible cases, deterministic and independent of LLM results.
            chosen.extend([(max(good, key=lambda r: r["errors"][0]), "trust"),
                           (min(bad, key=lambda r: r["errors"][0]), "reject")])
    random.Random(9062026).shuffle(chosen)
    manifest = {"budgets": BUDGETS, "environment": result["environment"], "cases": {}}
    for i, (r, label) in enumerate(chosen, 1):
        name = f"case_{i:02d}"
        cfg = {k:v for k,v in r["config"].items() if k != "pair"}
        folder = ROOT / "eval_cases" / name
        folder.mkdir(parents=True, exist_ok=True)
        code, contract = source(cfg), problem(cfg)
        (folder / "kernel.py").write_text(code)
        (folder / "problem.txt").write_text(contract)
        manifest["cases"][name] = {**r, "ground_truth": label,
            "kernel_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "problem_sha256": hashlib.sha256(contract.encode()).hexdigest()}
    (ROOT / "answer_key.json").write_text(json.dumps(manifest, indent=2))
    print("Built 24 GPU-validated cases, 4 trust + 4 reject per family.")
