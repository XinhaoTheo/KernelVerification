"""Numerical-only baseline helpers; no labels or private qualification inputs.

Modern cases use the public generator's default shape and seed zero once.
The original cohort retains its existing NAIVE_ALLCLOSE_VERDICT check.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import re
import sys
import tempfile
import time
import traceback

PROTOCOL_VERSION = "public_default_allclose_v1"
SEED = 0
VERDICT_RE = re.compile(r"NAIVE_ALLCLOSE_VERDICT:\s*(True|False)")


def comparison_profile(detail: dict) -> dict:
    """Select a numerical requirement by operation, never by case number/label."""
    cohort = detail.get("cohort")
    provenance = pathlib.Path(detail.get("provenance_file", "")).stem
    if provenance in {"state_pair_source", "state_contract_audit_source"}:
        return {"operation": "state_copy", "kind": "bits"}
    if provenance in {"norm_pair_source", "norm_contract_audit_source"}:
        return {"operation": "rmsnorm_backward", "kind": "rmsnorm"}
    if cohort == "boundary":
        if provenance == "segmented_state_source":
            return {"operation": "segmented_state", "kind": "allclose", "atol": .002, "rtol": .0001}
        if provenance == "paged_attention_source":
            return {"operation": "paged_attention", "kind": "allclose", "atol": .003, "rtol": .003}
    if cohort == "splitk_moe":
        cohort = "splitk" if provenance == "splitk_source" else "moe" if provenance == "moe_source" else None
    numeric = {
        "splitk": (.01, .001), "moe": (.01, .0002), "int4_pipeline": (.01, .001),
        "stencil": (2e-5, 0), "requant": (1e-6, 0), "erf": (1e-6, 0),
        "gelu": (1e-6, 0), "silu": (1e-6, 0), "sine": (1e-6, 0),
        "hypot": (1e-6, 0), "softplus": (1e-6, 0), "compensated": (1e-8, 1e-5),
        "atan": (1e-6, 0), "complex": (1e-7, 5e-7), "realfft": (1e-4, 1e-6),
        "exp2": (0, 1e-6), "log1p": (1e-6, 0),
    }
    if cohort in numeric:
        atol, rtol = numeric[cohort]
        return {"operation": cohort, "kind": "allclose", "atol": atol, "rtol": rtol}
    if cohort in {"fastdiv", "butterfly", "packed_argmax", "ntt", "crc", "requant_compose", "affine_scan", "winograd", "aes"}:
        return {"operation": cohort, "kind": "exact", "atol": 0, "rtol": 0}
    if cohort == "rounding":
        return {"operation": cohort, "kind": "bits"}
    raise ValueError(f"No reviewed public comparison profile for cohort={cohort!r}, provenance={provenance!r}")


def generator_defaults(source: str, name: str = "make_inputs") -> dict:
    """Read public default arguments without importing a CUDA module locally."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            pairs = zip(node.args.args[-len(node.args.defaults):], node.args.defaults)
            return {arg.arg: ast.literal_eval(value) for arg, value in pairs}
    raise ValueError(f"Missing {name} in public kernel")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_description(value):
    import torch
    if isinstance(value, torch.Tensor):
        saved = value.detach().cpu().contiguous()
        raw = saved.view(torch.uint8).numpy().tobytes()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                "stride": list(value.stride()), "storage_offset": value.storage_offset(),
                "numel": value.numel(), "sha256": sha256(raw)}
    if isinstance(value, (tuple, list)):
        return [tensor_description(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_description(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"python_type": type(value).__name__, "value": value}
    raise TypeError(f"Unsupported input/output type: {type(value)}")


def compare_outputs(actual, reference, profile: dict, inputs, path: str = "output") -> list[dict]:
    """Keep integer comparisons exact and perform floating checks in FP64."""
    import torch
    if isinstance(reference, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(reference):
            return [{"path": path, "passed": False, "reason": "output structure differs"}]
        if profile["kind"] == "rmsnorm":
            x, weight, dy, rstd = inputs
            h = (x.float() * rstd[:, None]).to(torch.float16)
            terms = (dy * h).to(torch.float64).abs().sum(dim=0)
            profiles = [dict(kind="allclose", atol=.002, rtol=.002),
                        dict(kind="scaled", absolute_limit=1e-5 + 1e-5 * terms)]
        else:
            profiles = [profile] * len(reference)
        return [row for i, (a, r, p) in enumerate(zip(actual, reference, profiles))
                for row in compare_outputs(a, r, p, inputs, f"{path}[{i}]")]
    if not isinstance(actual, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError("Baseline supports tensor outputs and tuples/lists of tensors")
    row = {"path": path, "actual": tensor_description(actual), "reference": tensor_description(reference)}
    if actual.shape != reference.shape:
        return [dict(row, passed=False, reason="output shape differs")]
    expected = reference.to(actual.device)
    kind = profile["kind"]
    if kind == "bits":
        if actual.dtype != reference.dtype:
            return [dict(row, passed=False, reason="bitwise comparison requires equal dtypes")]
        a = actual.contiguous().view(torch.uint8).reshape(actual.numel(), actual.element_size())
        r = expected.contiguous().view(torch.uint8).reshape(actual.numel(), actual.element_size())
        mismatch = (a != r).any(dim=1)
        return [dict(row, passed=not bool(mismatch.any()), mismatched_elements=int(mismatch.sum()), comparison="exact_bits")]
    if kind == "exact":
        # Never convert integers to FP64: doing so can hide differences above 2**53.
        mismatch = actual != expected
        return [dict(row, passed=not bool(mismatch.any()), mismatched_elements=int(mismatch.sum()), comparison="exact_values")]
    a, r = actual.double(), expected.double()
    error = (a - r).abs()
    limit = profile["absolute_limit"].to(a.device) if kind == "scaled" else profile["atol"] + profile["rtol"] * r.abs()
    valid = torch.isfinite(a) & torch.isfinite(r) & (error <= limit)
    ratio = error / limit
    finite_error = error[torch.isfinite(error)]
    finite_ratio = ratio[torch.isfinite(ratio)]
    row.update(passed=bool(valid.all()), mismatched_elements=int((~valid).sum()),
               comparison="abs(actual-reference)<=limit; nonfinite values fail",
               max_abs_error=float(finite_error.max()) if finite_error.numel() else None,
               max_error_ratio=float(finite_ratio.max()) if finite_ratio.numel() else None,
               nonfinite_actual=int((~torch.isfinite(a)).sum()),
               nonfinite_reference=int((~torch.isfinite(r)).sum()))
    if kind == "scaled":
        row["limit"] = "1e-5 + 1e-5 * sum(abs(rounded FP16 dy*h), dim=0)"
    else:
        row.update(atol=profile["atol"], rtol=profile["rtol"])
    return [row]


def execute_payload(payload: dict) -> dict:
    """Execute one public check; infrastructure/runtime failures are no verdict."""
    import random
    import numpy as np
    import torch
    import triton
    started = time.perf_counter()
    result = {"case": payload["case"], "protocol_version": PROTOCOL_VERSION,
              "mode": payload["mode"], "verdict": "no_verdict", "passed": None,
              "error": None, "phase": "environment", "seed": SEED}
    output = io.StringIO()
    module_name = f"allclose_{payload['case']}"
    old_kernel = sys.modules.pop("kernel", None)
    with tempfile.TemporaryDirectory(prefix="kv_allclose_") as directory:
        root = pathlib.Path(directory)
        for name, source in payload["sources"].items():
            (root / name).write_text(source)
        sys.path.insert(0, directory)
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                torch.cuda.init()
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                result["environment"] = {"python": sys.version, "torch": torch.__version__, "triton": triton.__version__,
                    "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                    "device_capability": list(torch.cuda.get_device_capability())}
                result["phase"] = "import"
                target = "test.py" if payload["mode"] == "legacy_naive" else "kernel.py"
                spec = importlib.util.spec_from_file_location(module_name, root / target)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                if payload["mode"] == "legacy_naive":
                    result["phase"] = "legacy_test"
                    mod.test_kernel()
                    torch.cuda.synchronize()
                    matches = VERDICT_RE.findall(output.getvalue())
                    if not matches or len(set(matches)) != 1:
                        raise RuntimeError(f"Expected one consistent NAIVE_ALLCLOSE_VERDICT; got {matches}")
                    result["passed"] = matches[-1] == "True"
                else:
                    result["phase"] = "generate_inputs"
                    inputs = mod.make_inputs(device="cuda", seed=SEED)
                    if not isinstance(inputs, (tuple, list)):
                        raise TypeError("make_inputs must return an argument tuple/list")
                    result["inputs"] = tensor_description(inputs)
                    result["phase"] = "reference"
                    reference = mod.reference(*inputs)
                    # Some targets require in-place mutation. Calculate and materialize the
                    # reference first, so run cannot change its source values afterwards.
                    def freeze(value):
                        return tuple(freeze(x) for x in value) if isinstance(value, (tuple, list)) else value.detach().clone()
                    reference = freeze(reference)
                    torch.cuda.synchronize()
                    result["phase"] = "run"
                    actual = mod.run(*inputs)
                    torch.cuda.synchronize()
                    result["phase"] = "compare"
                    result["checks"] = compare_outputs(actual, reference, payload["comparison"], inputs)
                    result["passed"] = all(row["passed"] for row in result["checks"])
                result["verdict"] = "trust" if result["passed"] else "reject"
                result["phase"] = "complete"
        except Exception:
            result["error"] = traceback.format_exc()
            result["verdict"] = "no_verdict"
            result["passed"] = None
        finally:
            sys.path.remove(directory)
            sys.modules.pop(module_name, None)
            sys.modules.pop("kernel", None)
            if old_kernel is not None:
                sys.modules["kernel"] = old_kernel
    result["stdout"] = output.getvalue()
    result["elapsed_seconds"] = time.perf_counter() - started
    return result
