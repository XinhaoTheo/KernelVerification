"""Private qualification utilities for finite-FP16 scalar mathematical targets.

This helper is versioned by its SHA256 in every qualification result. Existing
families retain their own frozen oracles; this file does not modify them.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import gmpy2
import mpmath as mp
import numpy as np
import torch

ORACLE_BITS = 320
MPMATH_DIGITS = 110


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def finite_bit_patterns():
    bits = np.arange(65536, dtype=np.uint16)
    finite = bits[(bits & 0x7C00) != 0x7C00]
    assert finite.size == 63488 and np.isfinite(finite.view(np.float16)).all()
    return finite


def build_reference_table(bounds, mp_target, operation, extra_metadata=None):
    """Outward MPFR intervals indexed by uint16 encoding, crosschecked by mpmath.

    bounds maps every finite bit pattern to a pair of finite MPFR endpoints.
    mp_target receives an exact mpmath value inside a 110-decimal context.
    """
    bits = finite_bit_patterns()
    lower = np.full(65536, np.nan, dtype=np.float64)
    upper, center = lower.copy(), lower.copy()
    with mp.workdps(MPMATH_DIGITS):
        for bit in bits:
            low, high = bounds[int(bit)]
            assert low <= high and gmpy2.is_finite(low) and gmpy2.is_finite(high)
            ln, ld = low.as_integer_ratio()
            hn, hd = high.as_integer_ratio()
            other = mp_target(mp.mpf(float(np.array([bit], dtype=np.uint16).view(np.float16)[0])))
            assert mp.mpf(int(ln)) / int(ld) <= other <= mp.mpf(int(hn)) / int(hd), (int(bit), str(low), str(high), str(other))
            center[bit] = float(other)
    with gmpy2.context(precision=53, round=gmpy2.RoundDown):
        for bit in bits:
            lower[bit] = float(gmpy2.mpfr(bounds[int(bit)][0], 53))
    with gmpy2.context(precision=53, round=gmpy2.RoundUp):
        for bit in bits:
            upper[bit] = float(gmpy2.mpfr(bounds[int(bit)][1], 53))
    # Reconvert binary64 endpoints exactly to check the outward conversion.
    with gmpy2.context(precision=ORACLE_BITS, round=gmpy2.RoundToNearest):
        for bit in bits:
            low, high = bounds[int(bit)]
            assert gmpy2.mpfr(float(lower[bit])) <= low
            assert gmpy2.mpfr(float(upper[bit])) >= high
    assert np.all(lower[bits] <= center[bits]) and np.all(center[bits] <= upper[bits])
    uncertainty = np.maximum(center[bits] - lower[bits], upper[bits] - center[bits])
    metadata = {
        "passed": True, "operation": operation, "finite_encodings": int(bits.size),
        "subnormal_encodings": int((((bits & 0x7C00) == 0) & ((bits & 0x3FF) != 0)).sum()),
        "signed_zero_encodings": int(((bits & 0x7FFF) == 0).sum()),
        "excluded_nan_inf_encodings": 65536 - int(bits.size),
        "mpfr_precision_bits": ORACLE_BITS, "mpmath_decimal_digits": MPMATH_DIGITS,
        "mpmath_independent_crosschecks": int(bits.size),
        "directed_binary64_endpoint_checks": 2 * int(bits.size),
        "gmpy2_version": gmpy2.version(), "mpfr_version": gmpy2.mpfr_version(),
        "mpmath_version": mp.__version__,
        "max_binary64_interval_width": float(np.max(upper[bits] - lower[bits])),
        "max_center_reference_error_bound": float(np.max(uncertainty)),
        "finite_bits_sha256": hashlib.sha256(bits.astype("<u2").tobytes()).hexdigest(),
        "lower_table_sha256": hashlib.sha256(lower[bits].astype("<f8").tobytes()).hexdigest(),
        "upper_table_sha256": hashlib.sha256(upper[bits].astype("<f8").tobytes()).hexdigest(),
        "oracle_dependency_hashes": dependency_hashes(),
        "scope": "Directed mathematical intervals supply the oracle; mpmath agreement is an independent numerical cross-check. Exhaustive scalar encodings are separate from tensor indexing and GPU specialization checks.",
    }
    metadata.update(extra_metadata or {})
    for array in (lower, upper, center):
        array.setflags(write=False)
    return lower, upper, center, metadata


def compare(observed, bits, table, atol):
    lower, upper, center, _ = table
    y = np.asarray(observed, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint16)
    assert y.shape == bits.shape and y.ndim == 1 and y.size > 0
    lo, hi, ref = lower[bits], upper[bits], center[bits]
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    finite = np.isfinite(y)
    worst = np.nextafter(np.maximum(np.abs(y - lo), np.abs(y - hi)), np.inf)
    best = np.maximum(np.nextafter(np.maximum(np.maximum(lo - y, y - hi), 0.0), -np.inf), 0.0)
    failing = (~finite) | (best > atol)
    uncertain = finite & (best <= atol) & (worst > atol)
    index = int(np.argmax(np.where(finite, worst, np.inf)))
    witness = None
    if failing.any() or uncertain.any():
        witness = {"index": index, "input_bits_hex": f"0x{int(bits[index]):04x}",
                   "input": float(bits[index:index + 1].view(np.float16)[0]),
                   "actual": float(y[index]), "reference": float(ref[index]),
                   "reference_lower": float(lo[index]), "reference_upper": float(hi[index]),
                   "absolute_error_lower_bound": float(best[index]),
                   "absolute_error_upper_bound": float(worst[index]), "allowed_error": atol}
    maximum = float(np.max(worst))
    return {"numeric_passed": bool(finite.all() and not failing.any() and not uncertain.any()),
            "finite": bool(finite.all()), "elements": int(y.size),
            "definite_failures": int(failing.sum()), "interval_ambiguous_elements": int(uncertain.sum()),
            "max_abs_error": float(np.max(np.abs(y - ref))), "max_abs_error_upper_bound": maximum,
            "max_error_ratio": maximum / atol, "minimum_tolerance_margin": atol - maximum,
            "worst_input_bits_hex": f"0x{int(bits[index]):04x}",
            "worst_input": float(bits[index:index + 1].view(np.float16)[0]), "witness": witness}


def public_reference_check(observed, bits, table, atol, reference_budget):
    assert observed.dtype in (np.float32, np.float64)
    lower, upper, _, _ = table
    rounded_low, rounded_high = lower[bits].astype(np.float32), upper[bits].astype(np.float32)
    unique = rounded_low == rounded_high
    certified = unique & (observed == rounded_low) if observed.dtype == np.float32 else np.zeros_like(unique)
    result = compare(observed, bits, table, atol)
    assert result["numeric_passed"] and result["max_abs_error_upper_bound"] <= reference_budget, result
    return {**result, "public_reference_dtype": str(observed.dtype),
            "fp32_unique_rounding_intervals": int(unique.sum()),
            "fp32_correct_rounding_certified_elements": int(certified.sum()),
            "interval_certified_fp32_rounding": bool(certified.all())}


def indexing_audit(block=256, max_n=131072):
    sizes = np.arange(1, max_n + 1, dtype=np.int64)
    grids = (sizes + block - 1) // block
    active_tail = sizes - (grids - 1) * block
    assert np.all((active_tail >= 1) & (active_tail <= block))
    assert int(np.max(grids * block - 1)) < 2**31
    return {"passed": True, "legal_lengths_checked": int(sizes.size), "block_size": block,
            "maximum_generated_index": int(np.max(grids * block - 1)),
            "argument": "For index=program_id*BLOCK+lane, adjacent blocks cover disjoint adjacent intervals. mask index<N retains [0,N) exactly once. The full legal N range has 1..BLOCK active lanes in its last block, with int32-safe address indices.",
            "scope": "Arithmetic model of the audited source indexing, not exhaustive GPU testing of every N, address, or compiler specialization."}


def plans(seed=128000):
    lengths = (1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257,
               511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097,
               63487, 63489, 65535, 65536, 65537, 131071, 131072)
    rows = [dict(n=n, style="random_bits", seed=seed + i, offset=0) for i, n in enumerate(lengths)]
    for i, style in enumerate(("all_finite", "all_finite_reverse", "all_finite_shuffle")):
        rows.append(dict(n=63488, style=style, seed=seed + 100 + i, offset=0))
    rows += [dict(n=63488, style="all_finite_shuffle", seed=seed + 110, offset=7),
             dict(n=257, style="boundary_bits", seed=seed + 111, offset=1),
             dict(n=131072, style="boundary_bits", seed=seed + 112, offset=13),
             dict(n=2046, style="subnormals", seed=seed + 113, offset=3),
             dict(n=129, style="signed_zeros", seed=seed + 114, offset=5)]
    return rows


def arrays(plan):
    all_bits = finite_bit_patterns()
    rng = np.random.default_rng(plan["seed"])
    style = plan["style"]
    if style.startswith("all_finite"):
        bits = all_bits.copy()
        if style.endswith("reverse"):
            bits = bits[::-1].copy()
        elif style.endswith("shuffle"):
            rng.shuffle(bits)
    elif style == "boundary_bits":
        positive = [0, 1, 2, 0x03FF, 0x0400, 0x0401, 0x2FFF, 0x3000, 0x3001,
                    0x37FF, 0x3800, 0x3801, 0x3BFF, 0x3C00, 0x3C01,
                    0x3FFF, 0x4000, 0x4001, 0x43FF, 0x4400, 0x4401,
                    0x47FF, 0x4800, 0x4801, 0x4BFF, 0x4C00, 0x4C01, 0x7BFE, 0x7BFF]
        bits = np.resize(np.array(positive + [b | 0x8000 for b in positive], dtype=np.uint16), plan["n"])
    elif style == "subnormals":
        bits = all_bits[((all_bits & 0x7C00) == 0) & ((all_bits & 0x03FF) != 0)]
    elif style == "signed_zeros":
        bits = np.resize(np.array([0, 0x8000], dtype=np.uint16), plan["n"])
    else:
        bits = rng.choice(all_bits, size=plan["n"], replace=True)
    assert bits.size == plan["n"] and bits.dtype == np.uint16
    return bits.view(np.float16).copy()


def public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"validate_inputs", "reference", "make_inputs"}]
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def cpu_audit(case_root, expected, table, atol, reference_budget):
    values = finite_bit_patterns().view(np.float16)
    checks = []
    for name in expected:
        public = public_cpu(case_root, name)
        tensor = torch.from_numpy(values.copy())
        public["validate_inputs"](tensor, require_cuda=False)
        result = public_reference_check(public["reference"](tensor).detach().cpu().numpy(), values.view(np.uint16), table, atol, reference_budget)
        checks.append({"case": name, **result})
    return {"passed": True, "oracle": table[3], "indexing": indexing_audit(),
            "public_references": checks, "oracle_dependency_hashes": dependency_hashes(),
            "scope": "Full finite-FP16 scalar reference audit, distinct from GPU arithmetic and tensor-property qualification."}


def _load(case_root, name, operation):
    modname = operation + "_qualification_" + name
    spec = importlib.util.spec_from_file_location(modname, Path(case_root) / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root, expected, table, operation, atol, reference_budget, seed=128000):
    cpu = cpu_audit(case_root, expected, table, atol, reference_budget)
    cases = {}
    for name, label in expected.items():
        module = _load(case_root, name, operation)
        records = []
        for plan in plans(seed):
            values, offset = arrays(plan), int(plan["offset"])
            backing = torch.full((offset + values.size + 23,), 0.3125, dtype=torch.float16, device="cuda")
            x = backing[offset:offset + values.size]
            x.copy_(torch.from_numpy(values).cuda())
            assert x.is_contiguous() and x.storage_offset() == offset
            before = backing.clone()
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            valid = type(actual) is torch.Tensor and actual.dtype == torch.float32 and actual.device == x.device and tuple(actual.shape) == tuple(x.shape)
            assert valid, (name, plan, "output type/dtype/device/shape")
            numeric = compare(actual.detach().cpu().numpy(), values.view(np.uint16), table, atol)
            preserved = bool(torch.equal(before.view(torch.uint8), backing.view(torch.uint8)))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            public = public_reference_check(module.reference(x).detach().cpu().numpy(), values.view(np.uint16), table, atol, reference_budget)
            records.append(dict(plan, legal=True, **numeric, passed=bool(numeric["numeric_passed"] and preserved and fresh),
                                inputs_and_backing_guards_preserved=preserved, inputs_preserved=preserved,
                                output_fresh=bool(fresh), shape_dtype_valid=True, storage_offset=int(x.storage_offset()),
                                public_reference_max_error_bound=public["max_abs_error_upper_bound"],
                                public_reference_interval_certified_fp32=public["interval_certified_fp32_rounding"],
                                unique_input_encodings=int(np.unique(values.view(np.uint16)).size),
                                exhaustive_scalar_domain=plan["style"].startswith("all_finite")))
        failures = [row for row in records if not row["passed"]]
        numeric_reject = any(row["definite_failures"] > 0 for row in records)
        cases[name] = {"ground_truth": label, "tests": records, "tests_total": len(records),
                       "tests_passed": len(records) - len(failures), "counterexamples": failures,
                       "qualification_passed": numeric_reject if label == "reject" else not failures and cpu["passed"],
                       "scalar_domain_encodings": 63488,
                       "exhaustive_scalar_passes": sum(row["exhaustive_scalar_domain"] and row["passed"] for row in records),
                       "gpu_elements_compared": sum(row["elements"] for row in records),
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One fixed implementation over the full public domain; changing inputs only supplies witnesses.",
                       "analysis": "Every finite FP16 encoding compared with a directed mathematical interval in several orders and alignments; other N/tails, guards, input preservation and fresh output checked separately. Scalar coverage alone is not every GPU specialization or memory execution."}
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": all(row["qualification_passed"] for row in cases.values())}
