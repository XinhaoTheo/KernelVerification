"""Independent qualification for case_108/109 (no model API calls).

Run validate(Path('/root/cases')) inside the frozen CUDA environment. Expected
labels are registered before GPU execution; test measurements only qualify or
invalidate that registration. They never relabel the same implementation.
"""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

EXPECTED = {"case_108": "trust", "case_109": "reject"}


def _load(path):
    spec = importlib.util.spec_from_file_location("norm_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(rows, cols, seed=0, style="random"):
    import torch
    generator = torch.Generator().manual_seed(seed)
    if style == "random":
        x = (2 * torch.rand((rows, cols), generator=generator) - 1).half()
        dy = (2 * torch.rand((rows, cols), generator=generator) - 1).half()
    elif style == "coherent":
        x = torch.ones((rows, cols), dtype=torch.float16)
        dy = torch.where(torch.arange(rows)[:, None] % 16 < 8, 1.0, 2.0**-12)
        dy = dy.expand(rows, cols).contiguous().half()
    elif style == "cancellation":
        x = torch.ones((rows, cols), dtype=torch.float16)
        signs = torch.where(torch.arange(rows) % 2 == 0, 1.0, -1.0)
        dy = signs[:, None].expand(rows, cols).contiguous().half()
        dy[-1] *= 0.5
    elif style in ("small_terms", "rms_lower_bound", "rms_upper_bound"):
        scale = {"small_terms": 1.0, "rms_lower_bound": .25, "rms_upper_bound": 4.0}[style]
        x = torch.full((rows, cols), scale, dtype=torch.float16)
        dy = torch.full_like(x, 2.0**-24 if style == "small_terms" else 4.0)
    elif style == "dynamic_range":
        signs = torch.where(torch.rand((rows, cols), generator=generator) < 0.5, -1., 1.)
        x = (signs * (0.25 + 3.75 * torch.rand((rows, cols), generator=generator))).half()
        dy = (signs * torch.pow(2., -12. * torch.rand((rows, cols), generator=generator))).half()
    else:
        raise ValueError(style)
    weight = (0.5 + torch.rand((cols,), generator=generator)).half()
    rstd = torch.rsqrt(x.float().square().mean(dim=1) + 1e-5)
    return x, weight, dy, rstd


def independent_reference(values):
    """CPU NumPy oracle, separate from candidate and public Torch reference."""
    import numpy as np
    x, w, dy, r = (v.cpu().numpy() for v in values)
    x64 = x.astype(np.float64)
    r64 = r.astype(np.float64)[:, None]
    m64 = (dy.astype(np.float32) * w.astype(np.float32)).astype(np.float16).astype(np.float64)
    h = (x.astype(np.float32) * r[:, None]).astype(np.float16)
    terms = (dy.astype(np.float32) * h.astype(np.float32)).astype(np.float16).astype(np.float64)
    dx = r64 * (m64 - x64 * (r64 * r64) * np.mean(m64 * x64, axis=1, keepdims=True))
    dw = np.sum(terms, axis=0, dtype=np.float64)
    limits = (0.002 + 0.002 * np.abs(dx), 1e-5 + 1e-5 * np.abs(terms).sum(axis=0))
    return (dx, dw), limits


def _legal(values):
    import numpy as np
    x, w, dy, r = (v.cpu().numpy() for v in values)
    rms = np.sqrt(np.mean(x.astype(np.float64)**2, axis=1))
    ideal_r = 1. / np.sqrt(np.mean(x.astype(np.float64)**2, axis=1) + 1e-5)
    return bool(
        np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(dy).all() and np.isfinite(r).all()
        and np.max(np.abs(x)) <= 4 and np.max(np.abs(dy)) <= 4 and np.max(np.abs(w)) <= 2
        and np.min(rms) >= .25 and np.max(rms) <= 4
        and np.max(np.abs(r.astype(np.float64) / ideal_r - 1)) <= 1e-6
    )


def validate(case_root: Path) -> dict:
    import numpy as np
    import torch
    import triton
    plans = [
        (1,16,0,"random"), (31,31,1,"random"), (511,128,2,"random"),
        (512,128,3,"random"), (513,127,4,"random"), (768,256,5,"random"),
        (1025,255,6,"random"), (4096,64,7,"random"), (512,257,8,"random"),
        (4096,512,9,"random"), (512,64,0,"coherent"),
        (513,127,0,"coherent"), (4096,256,0,"coherent"),
        (1025,127,0,"cancellation"), (768,128,10,"dynamic_range"),
        (4096,64,0,"small_terms"), (513,31,0,"rms_lower_bound"),
        (513,31,0,"rms_upper_bound"),
    ]
    report = {
        "family": "norm_pair", "expected_labels": EXPECTED,
        "scope": "full declared input domain, not per-input labels",
        "environment": {"torch": torch.__version__, "triton": triton.__version__,
                        "gpu": torch.cuda.get_device_name(),
                        "sms": torch.cuda.get_device_properties(0).multi_processor_count},
        "cases": {}, "independent_reference": "NumPy FP64 with explicit FP16 product rounding",
    }
    for name, expected in EXPECTED.items():
        path = case_root / name / "kernel.py"
        module = _load(path)
        result = {"expected_label": expected,
                  "kernel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "tests": []}
        for rows, cols, seed, style in plans:
            host = inputs(rows, cols, seed, style)
            assert _legal(host), (rows, cols, seed, style)
            references, limits = independent_reference(host)
            # Check the public reference rather than trusting it as the oracle.
            public = module.reference(*host)
            public_error = max(float(np.max(np.abs(t.numpy() - ref))) for t, ref in zip(public, references))
            assert public_error < 1e-8, (name, style, public_error)
            gpu = tuple(v.cuda() for v in host)
            before = tuple(v.clone() for v in gpu)
            ratios, abs_errors = [0.,0.], [0.,0.]
            is_finite = True
            unchanged = True
            shape_dtype = True
            repeats = 10 if style == "coherent" else 2
            for _ in range(repeats):
                actual = module.run(*gpu)
                torch.cuda.synchronize()
                shape_dtype = shape_dtype and actual[0].shape == (rows, cols) and actual[1].shape == (cols,)
                shape_dtype = shape_dtype and actual[0].dtype == torch.float16 and actual[1].dtype == torch.float32
                unchanged = unchanged and all(torch.equal(a, b) for a,b in zip(gpu, before))
                for index, (value, ref, limit) in enumerate(zip(actual, references, limits)):
                    array = value.double().cpu().numpy()
                    is_finite = is_finite and bool(np.isfinite(array).all())
                    ratios[index] = max(ratios[index], float(np.max(np.abs(array-ref)/limit)))
                    abs_errors[index] = max(abs_errors[index], float(np.max(np.abs(array-ref))))
            passed = is_finite and unchanged and shape_dtype and max(ratios) <= 1
            result["tests"].append({
                "shape": [rows, cols], "seed": seed, "style": style, "legal": True,
                "repetitions": repeats, "passed": bool(passed), "max_error_ratio": dict(zip(("dx","dw"),ratios)),
                "max_abs_error": dict(zip(("dx","dw"),abs_errors)),
                "inputs_preserved": unchanged, "shape_dtype_valid": shape_dtype,
                "finite": is_finite, "public_reference_max_difference": public_error,
            })
        failures = [x for x in result["tests"] if not x["passed"]]
        result["tests_passed"] = len(result["tests"])-len(failures)
        result["tests_total"] = len(result["tests"])
        result["counterexamples"] = failures
        result["qualified"] = not failures if expected == "trust" else bool(failures)
        result["qualification_passed"] = result["qualified"]
        result["ground_truth"] = expected if result["qualified"] else None
        report["cases"][name] = result
    report["qualified"] = all(x["qualified"] for x in report["cases"].values())
    return report
