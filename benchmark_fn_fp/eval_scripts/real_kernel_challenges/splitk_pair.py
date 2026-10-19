"""Independent qualification for the fixed split-K GEMM implementations.

Importing this file runs neither a model nor a GPU service. The public input
contract and tolerance were fixed before searching for numerical witnesses.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_43": "reject", "case_44": "trust"}
ATOL, RTOL = .01, .001


def plans():
    rows = []
    for index, k in enumerate((1, 15, 31, 32, 33, 63, 127, 128, 129, 255, 256, 257, 511, 512)):
        rows.append(dict(m=(1, 17, 31, 64)[index % 4], n=(1, 15, 33, 64)[index % 4],
                         k=k, split=(1, 2, 4, 8)[index % 4], style="random", seed=100 + index))
    for split in (1, 2, 4, 8):
        rows.extend([
            dict(m=3, n=17, k=512, split=split, style="cancellation", seed=200),
            dict(m=17, n=33, k=512, split=split, style="maximum", seed=201),
            dict(m=5, n=19, k=257, split=split, style="subnormal", seed=202),
            dict(m=3, n=5, k=512, split=split, style="subnormal_mixed", seed=206),
        ])
    rows.extend([
        dict(m=17, n=33, k=511, split=8, style="alternating", seed=203),
        dict(m=2, n=1, k=1, split=8, style="zero", seed=204),
        dict(m=31, n=17, k=127, split=4, style="storage_offset", seed=205),
    ])
    return rows


def arrays(plan):
    rng = np.random.default_rng(plan["seed"])
    m, n, k = plan["m"], plan["n"], plan["k"]
    a = rng.uniform(-.5, .5, (m, k)).astype(np.float16)
    b = rng.uniform(-.5, .5, (k, n)).astype(np.float16)
    bias = rng.uniform(-1, 1, n).astype(np.float32)
    if plan["style"] == "cancellation":
        a.fill(.5)
        b[:256].fill(.5)
        b[256:].fill(-.5)
        b[255].fill(.46875)
        b[511].fill(-.453125)
        bias.fill(0)
    elif plan["style"] == "maximum":
        a.fill(.5)
        b.fill(.5)
        bias.fill(1)
    elif plan["style"] == "subnormal":
        tiny = np.asarray([0., 2**-24, -2**-24, 2**-14 - 2**-24, -2**-14], np.float16)
        a[:] = tiny[np.arange(m * k).reshape(m, k) % len(tiny)]
        b[:] = tiny[np.arange(k * n).reshape(k, n) % len(tiny)]
        bias[:] = np.nextafter(np.float32(0), np.float32(1))
    elif plan["style"] == "alternating":
        a.fill(.5)
        b[:] = np.where(np.arange(k)[:, None] % 2, -.5, .5)
        bias.fill(-.25)
    elif plan["style"] == "subnormal_mixed":
        a.fill(2**-14 - 2**-24)
        b.fill(.5)
        bias.fill(0)
    elif plan["style"] == "zero":
        a.fill(0)
        b.fill(0)
        bias.fill(-0.)
    return a, b, bias, plan["split"]


def independent_reference(values):
    """Serial FP64 sum of outer products; no GEMM or split grouping."""
    a, b, bias, _ = values
    result = np.zeros((a.shape[0], b.shape[1]), np.float64)
    for inner in range(a.shape[1]):
        result += a[:, inner].astype(np.float64)[:, None] * b[inner].astype(np.float64)[None, :]
    return result + bias.astype(np.float64)[None, :]


def storage_rounding_model(values, half_scratch):
    """A diagnostic scratch-cast model, not a substitute for T4 execution."""
    a, b, bias, splits = values
    stride = ((a.shape[1] + 31) // 32 + splits - 1) // splits * 32
    result = np.zeros((a.shape[0], b.shape[1]), np.float32)
    partials = []
    for split in range(splits):
        lo, hi = split * stride, min((split + 1) * stride, a.shape[1])
        part = np.einsum("ik,kj->ij", a[:, lo:hi].astype(np.float64),
                         b[lo:hi].astype(np.float64), optimize=False)
        part = part.astype(np.float32)
        if half_scratch and splits > 1:
            part = part.astype(np.float16).astype(np.float32)
        result = result + part
        partials.append(part)
    return result + bias[None, :], partials


def _public_cpu(case_root, name):
    path = case_root / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"validate_inputs", "reference", "make_inputs"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def cpu_audit(case_root: Path):
    public = _public_cpu(case_root, "case_44")
    rows = []
    for plan in plans():
        values = arrays(plan)
        host = tuple(torch.from_numpy(v.copy()) for v in values[:3]) + (values[3],)
        public["validate_inputs"](*host, require_cuda=False)
        target = independent_reference(values)
        public_ref = public["reference"](*host).numpy()
        np_ref = np.einsum("ik,kj->ij", values[0].astype(np.float64),
                           values[1].astype(np.float64), optimize=False) + values[2]
        assert all(np.isfinite(ref).all() for ref in (target, public_ref, np_ref))
        error = max(float(np.max(np.abs(target-public_ref))), float(np.max(np.abs(target-np_ref))))
        assert error < 1e-11, (plan, error)
        cast_model, partials = storage_rounding_model(values, True)
        assert np.isfinite(cast_model).all()
        cast_error = np.abs(cast_model.astype(np.float64)-target)
        rows.append(dict(plan, reference_agreement_max_abs=error,
                         scratch_cast_model_error_ratio=float(np.max(cast_error/(ATOL+RTOL*np.abs(target)))),
                         scratch_cast_model_only=True))
    witness = dict(m=1, n=1, k=512, split=2, style="cancellation", seed=200)
    values = arrays(witness)
    target = independent_reference(values)
    low, parts = storage_rounding_model(values, True)
    high, _ = storage_rounding_model(values, False)
    assert target.item() == .0078125 and low.item() == .03125
    assert high.item() == target.item()
    assert abs(low.item()-target.item()) > ATOL+RTOL*abs(target.item())
    invalid = []
    for kind in ("bound", "nan", "dtype", "split", "bool_split", "grad", "negative_view",
                 "inference_tensor", "overlap", "noncontiguous", "oversized_k"):
        a = torch.zeros((2, 4), dtype=torch.float16)
        b = torch.zeros((4, 3), dtype=torch.float16)
        bias = torch.zeros(3, dtype=torch.float32)
        split = 2
        if kind == "bound": a[0, 0] = 1
        elif kind == "nan": a[0, 0] = float("nan")
        elif kind == "dtype": a = a.float()
        elif kind == "split": split = 3
        elif kind == "bool_split": split = True
        elif kind == "grad": a.requires_grad_(True)
        elif kind == "negative_view": a = torch._neg_view(a)
        elif kind == "inference_tensor":
            with torch.inference_mode(): a = a.clone()
        elif kind == "overlap":
            base = torch.zeros(20, dtype=torch.float16)
            a, b = base[:8].reshape(2,4), base[:12].reshape(4,3)
        elif kind == "noncontiguous": a = torch.zeros((4, 2), dtype=torch.float16).T
        elif kind == "oversized_k":
            a, b = torch.zeros((2,513), dtype=torch.float16), torch.zeros((513,3), dtype=torch.float16)
        try:
            public["validate_inputs"](a,b,bias,split,require_cuda=False)
        except ValueError as exc:
            invalid.append({"kind": kind, "rejected": True, "reason": str(exc)})
        else:
            raise AssertionError(f"Accepted invalid input: {kind}")
    return {"passed": True, "plans": rows, "invalid_inputs": invalid,
            "fixed_counterexample": {"plan": witness, "target": target.item(),
                "fp32_model": high.item(), "fp16_scratch_model": low.item(),
                "stored_fp16_partials": [p.item() for p in parts],
                "allowed_error": ATOL+RTOL*abs(target.item())},
            "method": "Serial FP64 outer products vs public FP64 torch.matmul vs NumPy FP64 einsum. Scratch model is explanatory only; GPU execution required."}


def _load(case_root, case):
    name = "splitk_qualification_" + case
    spec = importlib.util.spec_from_file_location(name, case_root / case / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root: Path):
    cpu = cpu_audit(case_root)
    cases = {}
    for name, expected in EXPECTED.items():
        module = _load(case_root, name)
        records = []
        for plan in plans():
            values = arrays(plan)
            ref = independent_reference(values)
            tensors = []
            for value in values[:3]:
                tensor = torch.from_numpy(value.copy()).cuda()
                if plan["style"] == "storage_offset":
                    base = torch.empty(tensor.numel()+13, dtype=tensor.dtype, device="cuda")
                    view = base[7:7+tensor.numel()].reshape(tensor.shape)
                    view.copy_(tensor)
                    tensor = view
                tensors.append(tensor)
            gpu = tuple(tensors) + (values[3],)
            legality = module.validate_inputs(*gpu)
            before = tuple(t.clone() for t in tensors)
            repetitions = 3 if plan["style"] == "cancellation" else 2
            ratio, max_abs = 0., 0.
            finite, preserved, types_valid = True, True, True
            witness = None
            for repeat in range(repetitions):
                actual = module.run(*gpu)
                torch.cuda.synchronize()
                public_ref = module.reference(*gpu).cpu().numpy()
                assert np.max(np.abs(public_ref-ref)) < 1e-11
                types_valid &= tuple(actual.shape) == ref.shape and actual.dtype == torch.float32 and actual.device == tensors[0].device
                preserved &= all(torch.equal(old,new) for old,new in zip(before,tensors))
                output = actual.double().cpu().numpy()
                finite &= bool(np.isfinite(output).all())
                delta = np.abs(output-ref)
                ratios = delta/(ATOL+RTOL*np.abs(ref))
                ratio = max(ratio, float(np.max(ratios)))
                max_abs = max(max_abs, float(np.max(delta)))
                if np.max(ratios) > 1 and witness is None:
                    index = np.unravel_index(np.argmax(ratios), ratios.shape)
                    witness = {"index": list(map(int,index)), "actual": float(output[index]),
                        "reference": float(ref[index]), "absolute_error": float(delta[index]),
                        "allowed_error": float(ATOL+RTOL*abs(ref[index])),
                        "ratio": float(ratios[index]), "repeat": repeat}
            records.append(dict(plan, legal=legality["valid"], repetitions=repetitions,
                passed=bool(finite and preserved and types_valid and ratio<=1),
                max_error_ratio=ratio, max_abs_error=max_abs, finite=finite,
                inputs_preserved=preserved, shape_dtype_valid=bool(types_valid), witness=witness))
        failures = [r for r in records if not r["passed"]]
        cases[name] = {"ground_truth": expected,
            "qualification_passed": bool(failures) if expected == "reject" else not failures,
            "tests": records, "tests_total": len(records), "tests_passed": len(records)-len(failures),
            "counterexamples": failures,
            "label_scope": "fixed implementation over the complete public domain",
            "analysis": "FP32 accumulation and final merge have conservative error below 0.01 in the bounded domain; no half partial can overflow. The rejected implementation has finite scratch-rounding cancellation counterexamples. Sampling is not a formal proof."}
    return {"cases": cases, "cpu_audit": cpu,
            "qualification_passed": all(c["qualification_passed"] for c in cases.values())}
