"""Independent directed-rounding erf oracle for case_53 and case_54.

Qualification dependencies: gmpy2==2.2.1 and mpmath==1.3.0. The MPFR library
version is recorded at runtime. Every finite FP16 *bit pattern* is included,
including both signed zeros and every subnormal. No candidate approximation
coefficient or arithmetic helper is used by the reference.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import gmpy2
import mpmath as mp
import numpy as np
import torch

EXPECTED = {"case_53": "trust", "case_54": "reject"}
ATOL = 1e-6
ORACLE_BITS = 320
MPMATH_DIGITS = 110


def finite_bit_patterns():
    """Reinterpret uint16 encodings; a numerical cast is not enumeration."""
    bits = np.arange(65536, dtype=np.uint16)
    finite = bits[(bits & 0x7C00) != 0x7C00]
    assert finite.size == 63488
    assert np.isfinite(finite.view(np.float16)).all()
    return finite


@lru_cache(maxsize=1)
def reference_table():
    positive_bits = np.arange(0x7C00, dtype=np.uint16)
    positive = positive_bits.view(np.float16)
    # FP16 -> binary64 -> MPFR is exact at 320-bit precision.
    with gmpy2.context(precision=ORACLE_BITS, round=gmpy2.RoundDown):
        lows = [gmpy2.erf(gmpy2.mpfr(float(value))) for value in positive]
    with gmpy2.context(precision=ORACLE_BITS, round=gmpy2.RoundUp):
        highs = [gmpy2.erf(gmpy2.mpfr(float(value))) for value in positive]
    crosschecks = 0
    centers = []
    # Independent erf implementation, with more decimal digits than the MPFR
    # interval precision. Exact power-of-two endpoint ratios avoid text errors.
    with mp.workdps(MPMATH_DIGITS):
        for value, lower, upper in zip(positive, lows, highs):
            ln, ld = lower.as_integer_ratio()
            un, ud = upper.as_integer_ratio()
            low_mp = mp.mpf(int(ln)) / int(ld)
            high_mp = mp.mpf(int(un)) / int(ud)
            other = mp.erf(mp.mpf(float(value)))
            assert low_mp <= other <= high_mp, (float(value), str(lower), str(upper), str(other))
            centers.append(float(other))
            crosschecks += 1
    # Convert enclosure endpoints to binary64 with directed rounding too.
    # float() then converts an exactly representable 53-bit value without loss.
    with gmpy2.context(precision=53, round=gmpy2.RoundDown):
        low64 = np.array([float(gmpy2.mpfr(value, 53)) for value in lows])
    with gmpy2.context(precision=53, round=gmpy2.RoundUp):
        high64 = np.array([float(gmpy2.mpfr(value, 53)) for value in highs])
    center64 = np.asarray(centers, dtype=np.float64)
    assert np.all(low64 <= center64) and np.all(center64 <= high64)
    lower = np.full(65536, np.nan, dtype=np.float64)
    upper = lower.copy()
    center = lower.copy()
    lower[positive_bits], upper[positive_bits], center[positive_bits] = low64, high64, center64
    negative_bits = positive_bits | np.uint16(0x8000)
    # erf is odd. Negation of dyadic bounds is exact and reverses endpoints.
    lower[negative_bits], upper[negative_bits], center[negative_bits] = -high64, -low64, -center64
    all_bits = finite_bit_patterns()
    subnormal = ((all_bits & 0x7C00) == 0) & ((all_bits & 0x03FF) != 0)
    zeros = (all_bits & 0x7FFF) == 0
    assert int(subnormal.sum()) == 2046 and int(zeros.sum()) == 2
    widths = upper[all_bits] - lower[all_bits]
    center_error = np.maximum(center[all_bits] - lower[all_bits], upper[all_bits] - center[all_bits])
    maximum_width = float(np.max(widths))
    assert maximum_width <= 2.0**-52
    metadata = {
        "passed": True, "finite_encodings": int(all_bits.size),
        "positive_or_zero_encodings": int(positive_bits.size),
        "negative_or_negative_zero_encodings": int(negative_bits.size),
        "subnormal_encodings": int(subnormal.sum()), "signed_zero_encodings": int(zeros.sum()),
        "excluded_nan_inf_encodings": 65536 - int(all_bits.size),
        "mpfr_precision_bits": ORACLE_BITS, "mpmath_decimal_digits": MPMATH_DIGITS,
        "mpmath_independent_crosschecks": crosschecks,
        "gmpy2_version": gmpy2.version(), "mpfr_version": gmpy2.mpfr_version(),
        "mpmath_version": mp.__version__,
        "max_binary64_interval_width": maximum_width,
        "max_center_reference_error_bound": float(np.max(center_error)),
        "finite_bits_sha256": hashlib.sha256(all_bits.astype("<u2").tobytes()).hexdigest(),
        "lower_table_sha256": hashlib.sha256(lower[all_bits].astype("<f8").tobytes()).hexdigest(),
        "upper_table_sha256": hashlib.sha256(upper[all_bits].astype("<f8").tobytes()).hexdigest(),
        "method": "320-bit MPFR erf under RoundDown and RoundUp, then outward binary64 endpoint conversion; 110-decimal mpmath erf crosschecks every nonnegative finite FP16 value; exact odd symmetry supplies negative bounds. No tail approximation is used.",
        "scope": "The directed-rounding interval is the numerical oracle. Independent mpmath agreement is a cross-check, not the source of the enclosure guarantee. Exhausting scalar FP16 encodings does not alone prove arbitrary tensor indexing, shape specialization, input preservation or fresh-output behavior."
    }
    for array in (lower, upper, center):
        array.setflags(write=False)
    return lower, upper, center, metadata


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all()
    return reference_table()[2][array.view(np.uint16)]


def _compare(observed, bits):
    lower, upper, center, _ = reference_table()
    y = np.asarray(observed, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint16)
    lo, hi, ref = lower[bits], upper[bits], center[bits]
    finite = np.isfinite(y)
    # Outward final subtraction ensures these are conservative bounds on
    # |candidate - real erf|, not just distances to a rounded reference point.
    worst = np.maximum(np.abs(y - lo), np.abs(y - hi))
    worst = np.nextafter(worst, np.inf)
    best = np.maximum(np.maximum(lo - y, y - hi), 0.0)
    best = np.maximum(np.nextafter(best, -np.inf), 0.0)
    failing = (~finite) | (best > ATOL)
    uncertain = finite & (best <= ATOL) & (worst > ATOL)
    i = int(np.argmax(np.where(finite, worst, np.inf)))
    witness = None
    if failing.any() or uncertain.any():
        witness = {"index": i, "input_bits_hex": f"0x{int(bits[i]):04x}",
                   "input": float(bits[i:i+1].view(np.float16)[0]),
                   "actual": float(y[i]), "reference": float(ref[i]),
                   "reference_lower": float(lo[i]), "reference_upper": float(hi[i]),
                   "absolute_error_lower_bound": float(best[i]),
                   "absolute_error_upper_bound": float(worst[i]), "allowed_error": ATOL}
    maximum = float(np.max(worst))
    return {"numeric_passed": bool(finite.all() and not failing.any() and not uncertain.any()),
            "finite": bool(finite.all()), "elements": int(y.size),
            "definite_failures": int(failing.sum()), "interval_ambiguous_elements": int(uncertain.sum()),
            "max_abs_error": float(np.max(np.abs(y - ref))),
            "max_abs_error_upper_bound": maximum, "max_error_ratio": maximum / ATOL,
            "minimum_tolerance_margin": ATOL - maximum,
            "worst_input_bits_hex": f"0x{int(bits[i]):04x}",
            "worst_input": float(bits[i:i+1].view(np.float16)[0]), "witness": witness}


def _public_reference_check(observed, bits):
    lower, upper, _, _ = reference_table()
    low_rounded = lower[bits].astype(np.float32)
    high_rounded = upper[bits].astype(np.float32)
    # Both enclosure endpoints round to the same FP32 result, establishing
    # the unique round-to-nearest FP32 reference on the checked inputs.
    assert np.array_equal(low_rounded, high_rounded), "reference interval spans an FP32 rounding boundary"
    assert observed.dtype == np.float32 and np.array_equal(observed, low_rounded), "public reference is not the interval-certified FP32 result"
    result = _compare(observed, bits)
    assert result["max_abs_error_upper_bound"] <= 2.0**-25 + 2.0**-52
    return {**result, "interval_certified_fp32_rounding": True}


def indexing_audit():
    sizes = np.arange(1, 131073, dtype=np.int64)
    grids = (sizes + 255) // 256
    last_start = (grids - 1) * 256
    active_tail = sizes - last_start
    assert np.all((active_tail >= 1) & (active_tail <= 256))
    assert int(np.max(grids * 256 - 1)) == 131071
    return {"passed": True, "legal_lengths_checked": int(sizes.size), "block_size": 256,
            "maximum_generated_index": 131071,
            "argument": "For the source index=program_id*256+lane, consecutive blocks cover adjacent disjoint integer intervals. For every legal N, the last block has 1..256 active lanes; mask index<N leaves exactly [0,N) once. All index arithmetic fits int32. Each lane reads one FP16 value and writes one independent FP32 output; N affects the mask, not scalar arithmetic.",
            "scope": "Exhaustive N check of the source indexing model plus its interval-partition argument. It is not an exhaustive GPU execution over every N, lane placement, address, or compiler specialization."}


def plans():
    lengths = (1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257,
               511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097,
               63487, 63489, 65535, 65536, 65537, 131071, 131072)
    records = [dict(n=n, style="random_bits", seed=126000 + i, offset=0)
               for i, n in enumerate(lengths)]
    for i, style in enumerate(("all_finite", "all_finite_reverse", "all_finite_shuffle")):
        records.append(dict(n=63488, style=style, seed=127000 + i, offset=0))
    records += [dict(n=63488, style="all_finite_shuffle", seed=127010, offset=7),
                dict(n=257, style="boundary_bits", seed=127011, offset=1),
                dict(n=131072, style="boundary_bits", seed=127012, offset=13),
                dict(n=2046, style="subnormals", seed=127013, offset=3),
                dict(n=129, style="signed_zeros", seed=127014, offset=5)]
    return records


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
        # Both signs, zero/subnormal/normal transitions, erf's active range,
        # and maximum finite input; none depend on candidate approximation.
        positive = [0, 1, 2, 0x03FF, 0x0400, 0x0401, 0x2FFF, 0x3000, 0x3001,
                    0x37FF, 0x3800, 0x3801, 0x3BFF, 0x3C00, 0x3C01,
                    0x3FFF, 0x4000, 0x4001, 0x43FF, 0x4400, 0x4401,
                    0x47FF, 0x4800, 0x4801, 0x7BFE, 0x7BFF]
        bits = np.resize(np.array(positive + [b | 0x8000 for b in positive], dtype=np.uint16), plan["n"])
    elif style == "subnormals":
        bits = all_bits[((all_bits & 0x7C00) == 0) & ((all_bits & 0x03FF) != 0)]
    elif style == "signed_zeros":
        bits = np.resize(np.array([0, 0x8000], dtype=np.uint16), plan["n"])
    else:
        bits = rng.choice(all_bits, size=plan["n"], replace=True)
    assert bits.size == plan["n"] and bits.dtype == np.uint16
    return bits.view(np.float16).copy()


def _public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"validate_inputs", "reference", "make_inputs"}]
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def cpu_audit(case_root):
    _, _, _, metadata = reference_table()
    values = finite_bit_patterns().view(np.float16)
    public_checks = []
    for name in EXPECTED:
        public = _public_cpu(case_root, name)
        tensor = torch.from_numpy(values.copy())
        public["validate_inputs"](tensor, require_cuda=False)
        observed = public["reference"](tensor).detach().cpu().numpy()
        result = _public_reference_check(observed, values.view(np.uint16))
        public_checks.append({"case": name, **result})
    return {"passed": True, "oracle": metadata, "indexing": indexing_audit(), "public_references": public_checks,
            "scope": "Exhaustive scalar input/reference validation; candidate GPU arithmetic and tensor properties are checked separately."}


def _load(case_root, name):
    modname = "erf_qualification_" + name
    spec = importlib.util.spec_from_file_location(modname, Path(case_root) / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root: Path):
    cpu = cpu_audit(case_root)
    cases = {}
    for name, expected_label in EXPECTED.items():
        module = _load(case_root, name)
        records = []
        for plan in plans():
            values = arrays(plan)
            offset = int(plan["offset"])
            # Every call has backing guards, even when storage_offset is zero.
            backing = torch.full((offset + values.size + 23,), 0.3125, dtype=torch.float16, device="cuda")
            x = backing[offset:offset + values.size]
            x.copy_(torch.from_numpy(values).cuda())
            assert x.is_contiguous() and x.storage_offset() == offset
            before = backing.clone()
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            valid_type = (type(actual) is torch.Tensor and actual.dtype == torch.float32
                          and actual.device == x.device and tuple(actual.shape) == tuple(x.shape))
            assert valid_type, (name, plan, "output shape/dtype/device")
            observed = actual.detach().cpu().numpy()
            numeric = _compare(observed, values.view(np.uint16))
            preserved = bool(torch.equal(before.view(torch.uint8), backing.view(torch.uint8)))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            public_result = _public_reference_check(module.reference(x).detach().cpu().numpy(), values.view(np.uint16))
            records.append(dict(plan, legal=True, **numeric,
                                passed=bool(numeric["numeric_passed"] and preserved and fresh and valid_type),
                                inputs_and_backing_guards_preserved=preserved, inputs_preserved=preserved,
                                output_fresh=bool(fresh), shape_dtype_valid=bool(valid_type),
                                storage_offset=int(x.storage_offset()),
                                public_reference_max_error_bound=public_result["max_abs_error_upper_bound"],
                                public_reference_interval_certified_fp32=True,
                                unique_input_encodings=int(np.unique(values.view(np.uint16)).size),
                                exhaustive_scalar_domain=plan["style"].startswith("all_finite")))
        failures = [record for record in records if not record["passed"]]
        certified_numeric_reject = any(record["definite_failures"] > 0 for record in records)
        cases[name] = {"ground_truth": expected_label, "tests": records,
                       "tests_total": len(records), "tests_passed": len(records) - len(failures),
                       "counterexamples": failures,
                       "qualification_passed": certified_numeric_reject if expected_label == "reject" else not failures and cpu["passed"],
                       "scalar_domain_encodings": 63488,
                       "exhaustive_scalar_passes": sum(record["exhaustive_scalar_domain"] and record["passed"] for record in records),
                       "gpu_elements_compared": sum(record["elements"] for record in records),
                       "label_scope": "One fixed implementation over all finite FP16 scalar values and the complete public tensor contract; inputs only provide witnesses.",
                       "analysis": "Every finite FP16 encoding is compared against a directed-rounding mathematical erf interval in multiple orders and storage alignments. Other lengths, tails, input preservation, backing guards and fresh storage are checked separately. Scalar enumeration does not by itself prove every shape specialization or compiler/memory behavior."}
    return {"cases": cases, "cpu_audit": cpu,
            "qualification_passed": all(case["qualification_passed"] for case in cases.values())}
