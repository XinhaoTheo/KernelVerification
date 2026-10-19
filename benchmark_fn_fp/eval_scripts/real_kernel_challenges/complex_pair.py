"""Exact-rational qualification for finite-FP16 complex division, cases154/155.

The oracle uses arbitrary-precision integers and rational division. It does not
use the candidate's rounded component ratio. The control has a whole-domain
roundoff argument; GPU plans separately check values and tensor behavior.
"""
from __future__ import annotations

import ast
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

import gmpy2
import numpy as np
import torch

EXPECTED = {"case_81": "trust", "case_82": "reject"}
ATOL, RTOL = 1e-7, 5e-7
# Decimal tolerances in the problem are exact real constants.
ATOL_Q, RTOL_Q = gmpy2.mpq(1, 10**7), gmpy2.mpq(5, 10**7)


def dependency_hashes():
    p = Path(__file__)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()}


def _ratios(numerator, denominator):
    """Every FP16 value times2^24 is an exact integer; scales cancel."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    assert numerator.dtype == denominator.dtype == np.float16
    assert numerator.shape == denominator.shape and numerator.ndim == 2
    assert numerator.shape[1] == 2 and np.isfinite(numerator).all() and np.isfinite(denominator).all()
    assert np.all(np.any(denominator != 0, axis=1))
    integers_x = (numerator.astype(np.float64) * (1 << 24)).astype(np.int64)
    integers_y = (denominator.astype(np.float64) * (1 << 24)).astype(np.int64)
    result = []
    for xy, zw in zip(integers_x, integers_y):
        a, b, c, d = map(int, (*xy, *zw))
        norm = c * c + d * d
        result.append((gmpy2.mpq(a * c + b * d, norm), gmpy2.mpq(b * c - a * d, norm)))
    return result


def independent_reference(numerator, denominator):
    return np.array([[float(r), float(i)] for r, i in _ratios(numerator, denominator)], dtype=np.float64)


def arithmetic_audit():
    u = Fraction(1, 1 << 24)
    bound = (1 + u)**2 / (1 - u) - 1
    lower_bound = 1 - (1 - u)**2 / (1 + u)
    assert lower_bound < bound < Fraction(5, 10**7)
    assert Fraction(1, 1 << 81) > Fraction(1, 1 << 126)
    assert (1 << 81) < (1 << 127)
    return {"passed": True, "unit_roundoff": float(u), "relative_error_upper_bound": float(bound),
            "relative_error_bound_exact": str(bound), "allowed_relative_error": RTOL,
            "proof": [
                "Every finite binary16 value is an integer multiple of2^-24 and has at most11 significant binary bits. Its product with another binary16 value has at most22 significant bits, so conversion and product in FP32 are exact.",
                "Every nonzero product is at least2^-48; all products have magnitude below2^32. A sum or difference of two products is either zero or at least2^-48, and its magnitude is below2^33. All exact and rounded nonzero products and sums are normal FP32 values; overflow and underflow do not occur.",
                "For either component, write the exact numerator asP and the positive exact denominator asD. The direct kernel computes RN(RN(P)/RN(D)), with exactly formed products and with implicit FMA disabled. Thus the relative factor is(1+e1)*(1+e3)/(1+e2), where each|ei|<=u. A zeroP is computed exactly aszero.",
                "The nonzero quotient's loose magnitude bounds2^-81 to2^81 remain inside the normal FP32 range, including the small roundoff factors. The maximum relative error is at most(1+u)^2/(1-u)-1<1.789e-7<5e-7. This proves the componentwise tolerance for every legal tuple, including severe cancellation.",
                "The scalar proof is bound to the candidate source expressions. GPU tests check actual lowering, tails, offsets, input guards and separate storage; this is not a formal proof of all GPU execution behavior."
            ]}


def _source_binding(case_root):
    result = {}
    expected = {"scale": "c * c + d * d", "real_numerator": "a * c + b * d",
                "imag_numerator": "b * c - a * d", "real": "tl.div_rn(real_numerator, scale)",
                "imag": "tl.div_rn(imag_numerator, scale)"}
    for name in EXPECTED:
        path = Path(case_root) / name / "kernel.py"
        tree = ast.parse(path.read_text())
        kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_divide")
        assignments = {node.targets[0].id: ast.unparse(node.value) for node in kernel.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        if name == "case_81":
            for target, expression in expected.items():
                assert assignments[target] == expression, (target, assignments[target])
        else:
            assert assignments["ratio"] == "tl.div_rn(minor, major)"
            assert "tl.fma(b, ratio, a)" in assignments["real_numerator"]
        for component in "abcd":
            assert assignments[component].endswith(".to(tl.float32)")
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert any(any(k.arg == "enable_fp_fusion" and isinstance(k.value, ast.Constant) and k.value.value is False
                       for k in node.keywords) for node in calls)
        result[name] = {"kernel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "expressions": {k: assignments[k] for k in expected},
                        "implicit_fma_disabled": True}
    return result


def plans():
    lengths = [1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257,
               511, 512, 513, 1023, 1024, 1025, 4093, 65537, 131072]
    rows = [dict(n=n, style="random", seed=154000 + i, offset_x=i % 7, offset_y=(i + 3) % 11)
            for i, n in enumerate(lengths)]
    rows += [dict(n=1, style="witness", seed=154100, offset_x=13, offset_y=17),
             dict(n=4097, style="cancellation", seed=154101, offset_x=3, offset_y=5),
             dict(n=4093, style="corners", seed=154102, offset_x=7, offset_y=9),
             dict(n=257, style="alias", seed=154103, offset_x=1, offset_y=1)]
    rows += [dict(n=63488, style="sweep_" + component, seed=154110 + i, offset_x=3, offset_y=5)
             for i, component in enumerate("abcd")]
    return rows


def _finite_values():
    bits = np.arange(65536, dtype=np.uint16)
    return bits[(bits & 0x7C00) != 0x7C00].view(np.float16)


def arrays(plan):
    n, style = int(plan["n"]), plan["style"]
    rng = np.random.default_rng(plan["seed"])
    finite = _finite_values()
    values = rng.choice(finite, size=(n, 4)).astype(np.float16)
    if style == "witness":
        values[:] = [1024, -1536, 3, 2]
    elif style == "cancellation":
        # These are generated legal witnesses, never an input-domain restriction.
        k = rng.integers(-10, 14, size=n)
        j = rng.integers(-10, 10, size=n)
        values[:, 0] = np.ldexp(np.ones(n), k)
        values[:, 1] = -1.5 * values[:, 0].astype(np.float32)
        values[:, 2] = np.ldexp(np.full(n, 3.0), j)
        values[:, 3] = np.ldexp(np.full(n, 2.0), j)
        swap = np.arange(n) % 2 == 1
        values[swap] = values[swap][:, [1, 0, 3, 2]]
    elif style == "corners":
        import itertools
        edge = [0.0, -0.0, 2**-24, -2**-24, 2**-14, -2**-14, 1.0, -1.0, 65504.0, -65504.0]
        corners = np.array([p for p in itertools.product(edge, repeat=4) if p[2] != 0 or p[3] != 0], dtype=np.float16)
        choice = rng.choice(corners.shape[0], size=n, replace=False)
        values = corners[choice]
    elif style.startswith("sweep_"):
        values[:] = [1024, -1536, 3, 2]
        values[:, "abcd".index(style[-1])] = finite
    zero = np.all(values[:, 2:] == 0, axis=1)
    values[zero, 2] = 1
    x, y = values[:, :2].copy(), values[:, 2:].copy()
    if style == "alias":
        x = y.copy()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x, y


def _public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"validate_inputs", "reference", "make_inputs"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def cpu_audit(case_root):
    proof = arithmetic_audit()
    source = _source_binding(case_root)
    plan = dict(n=513, style="random", seed=154099, offset_x=0, offset_y=0)
    x, y = arrays(plan)
    x[0], y[0] = [1024, -1536], [3, 2]
    exact = _ratios(x, y)
    checks = []
    for name in EXPECTED:
        public = _public_cpu(case_root, name)
        actual = public["reference"](torch.from_numpy(x), torch.from_numpy(y)).numpy()
        for pair, expected in zip(actual, exact):
            for value, target in zip(pair, expected):
                error = abs(gmpy2.mpq(float(value)) - target)
                assert error <= gmpy2.mpq(1, 1 << 50) * abs(target), (value, target)
        checks.append({"case": name, "rows": len(x), "public_reference_within_2_pow_minus_50_relative": True})
    # MPFR at24bits computes precisely the intended FP32 RN quotient/FMA witness.
    with gmpy2.context(precision=24, round=gmpy2.RoundToNearest):
        a, b, c, d = map(gmpy2.mpfr, [1024, -1536, 3, 2])
        ratio = d / c
        scale = gmpy2.fma(d, ratio, c)
        real = gmpy2.fma(b, ratio, a) / scale
        imag = gmpy2.fma(-a, ratio, b) / scale
    assert exact[0][0] == 0 and abs(gmpy2.mpq(real)) > ATOL_Q
    return {"passed": True, "arithmetic": proof, "source_binding": source,
            "public_reference_checks": checks,
            "cpu_rounding_witness": {"numerator": [1024, -1536], "denominator": [3, 2],
                                     "modeled_actual": [float(real), float(imag)],
                                     "exact_reference": list(map(str, exact[0])),
                                     "rounded_ratio": float(ratio), "actual_gpu_evidence": False},
            "oracle": "Exact arbitrary-precision rational components from integer-scaled FP16 values. Comparisons use exact rational error and exact decimal tolerances.",
            "indexing": "Program p covers indices256*p through256*p+255 with index<N masks. Each legal index reads2components from each input and writes2distinct output components. Maximum offset262143 is int32-safe. Nonzero input storage offsets are incorporated by the tensor pointer.",
            "oracle_dependency_hashes": dependency_hashes()}


def _load(case_root, name):
    spec = importlib.util.spec_from_file_location("complex_qualification_" + name, Path(case_root) / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu = cpu_audit(case_root)
    cases = {}
    prepared = [(plan, *arrays(plan)) for plan in plans()]
    oracle = [_ratios(x, y) for _, x, y in prepared]
    for name, label in EXPECTED.items():
        module = _load(case_root, name)
        records = []
        for (plan, x_np, y_np), targets in zip(prepared, oracle):
            n = plan["n"]
            ox, oy = plan["offset_x"], plan["offset_y"]
            bx = torch.full((ox + 2*n + 31,), 42, dtype=torch.float16, device="cuda")
            by = torch.full((oy + 2*n + 31,), -42, dtype=torch.float16, device="cuda")
            x, y = bx[ox:ox+2*n].view(n, 2), by[oy:oy+2*n].view(n, 2)
            x.copy_(torch.from_numpy(x_np).cuda())
            y.copy_(torch.from_numpy(y_np).cuda())
            if plan["style"] == "alias":
                y, by = x, bx
            before_x, before_y = bx.clone(), by.clone()
            actual = module.run(x, y)
            torch.cuda.synchronize()
            assert type(actual) is torch.Tensor and actual.dtype == torch.float32
            assert actual.device == x.device and tuple(actual.shape) == (n, 2)
            observed = actual.cpu().numpy()
            public = module.reference(x, y).cpu().numpy()
            failures = 0
            max_error, max_ratio, witness = 0.0, 0.0, None
            for row, (pair, target_pair) in enumerate(zip(observed, targets)):
                for component, (value, target) in enumerate(zip(pair, target_pair)):
                    public_error = abs(gmpy2.mpq(float(public[row, component])) - target)
                    assert public_error <= gmpy2.mpq(1, 1 << 50) * abs(target), (name, plan, "reference")
                    allowed = ATOL_Q + RTOL_Q * abs(target)
                    finite = np.isfinite(value)
                    error = abs(gmpy2.mpq(float(value)) - target) if finite else None
                    bad = not finite or error > allowed
                    max_error = max(max_error, float(error) if finite else float("inf"))
                    max_ratio = max(max_ratio, float(error / allowed) if finite else float("inf"))
                    if bad:
                        failures += 1
                        if witness is None:
                            witness = {"index": row, "component": "real" if component == 0 else "imaginary",
                                       "numerator": x_np[row].astype(float).tolist(),
                                       "denominator": y_np[row].astype(float).tolist(),
                                       "actual": float(value), "reference": float(target),
                                       "reference_exact_rational": str(target),
                                       "absolute_error": float(error) if finite else None,
                                       "allowed_error": float(allowed)}
            preserved = torch.equal(before_x.view(torch.uint8), bx.view(torch.uint8)) and torch.equal(before_y.view(torch.uint8), by.view(torch.uint8))
            fresh = all(actual.untyped_storage().data_ptr() != value.untyped_storage().data_ptr() for value in (x, y))
            records.append(dict(plan, passed=bool(failures == 0 and preserved and fresh), numeric_passed=failures == 0,
                                legal=True, definite_failures=failures, elements=n*2, rows=n,
                                max_abs_error=max_error, max_error_tolerance_ratio=max_ratio,
                                inputs_and_backing_guards_preserved=bool(preserved), output_fresh=bool(fresh),
                                shape_dtype_valid=True, witness=witness))
        failing = [row for row in records if not row["passed"]]
        numeric_reject = any(row["definite_failures"] for row in records)
        cases[name] = {"ground_truth": label, "tests": records, "tests_total": len(records),
                       "tests_passed": len(records) - len(failing), "counterexamples": failing,
                       "qualification_passed": bool(numeric_reject if label == "reject" else not failing and cpu["passed"]),
                       "atol": ATOL, "rtol": RTOL, "comparison": "exact-rational componentwise error against exact decimal tolerances",
                       "gpu_components_compared": sum(row["elements"] for row in records),
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One fixed implementation under the complete finite-FP16 complex input contract.",
                       "analysis": "Direct control has a source-bound whole-domain roundoff proof. Actual T4 checks use exact-rational references, random tuples, cancellation, component sweeps, extremes, tails, offsets, aliased inputs and guard preservation. Component sweeps do not enumerate all input tuples."}
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": all(row["qualification_passed"] for row in cases.values())}
