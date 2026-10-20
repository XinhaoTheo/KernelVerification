"""Independent mathematical expm1 qualification for case_97 and case_98."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import gmpy2
import mpmath as mp
import numpy as np
import torch

try:
    from . import adaptive_unary_oracle as core
except ImportError:
    import adaptive_unary_oracle as core

EXPECTED = {"case_97": "reject", "case_98": "trust"}
ATOL = 1e-7
RTOL = 1e-6
REFERENCE_BUDGET = 1e-10


def dependency_hashes():
    return {**core.dependency_hashes(),
            Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def legal_bits():
    bits = core.finite_bit_patterns()
    bits = bits[np.abs(bits.view(np.float16)) <= 10.0]
    assert bits.size == 37378
    return bits


@lru_cache(maxsize=1)
def reference_table():
    """Direct outward MPFR expm1, independent of candidate range reduction and polynomial."""
    bits = legal_bits()
    lower = np.full(65536, np.nan, dtype=np.float64)
    upper, center = lower.copy(), lower.copy()
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    with mp.workdps(core.MPMATH_DIGITS):
        for bit in bits:
            value = float(np.array([bit], dtype=np.uint16).view(np.float16)[0])
            with down:
                low = gmpy2.expm1(gmpy2.mpfr(value))
            with up:
                high = gmpy2.expm1(gmpy2.mpfr(value))
            assert low <= high and gmpy2.is_finite(low) and gmpy2.is_finite(high)
            other = mp.expm1(mp.mpf(value))
            ln, ld = low.as_integer_ratio()
            hn, hd = high.as_integer_ratio()
            assert mp.mpf(int(ln)) / int(ld) <= other <= mp.mpf(int(hn)) / int(hd)
            center[bit] = float(other)
            with gmpy2.context(precision=53, round=gmpy2.RoundDown):
                lower[bit] = float(gmpy2.mpfr(low, 53))
            with gmpy2.context(precision=53, round=gmpy2.RoundUp):
                upper[bit] = float(gmpy2.mpfr(high, 53))
            with gmpy2.context(precision=core.ORACLE_BITS):
                assert gmpy2.mpfr(float(lower[bit])) <= low
                assert gmpy2.mpfr(float(upper[bit])) >= high
    assert np.all(lower[bits] <= center[bits]) and np.all(center[bits] <= upper[bits])
    metadata = {
        "passed": True, "operation": "mathematical_expm1",
        "legal_scalar_encodings": int(bits.size),
        "domain": "Every finite FP16 encoding with value in [-10,10].",
        "subnormal_encodings": int((((bits & 0x7C00) == 0) & ((bits & 0x3FF) != 0)).sum()),
        "signed_zero_encodings": int(((bits & 0x7FFF) == 0).sum()),
        "mpfr_precision_bits": core.ORACLE_BITS,
        "mpmath_decimal_digits": core.MPMATH_DIGITS,
        "mpmath_independent_crosschecks": int(bits.size),
        "directed_binary64_endpoint_checks": 2 * int(bits.size),
        "gmpy2_version": gmpy2.version(), "mpfr_version": gmpy2.mpfr_version(),
        "mpmath_version": mp.__version__,
        "max_binary64_interval_width": float(np.max(upper[bits] - lower[bits])),
        "max_center_reference_error_bound": float(np.max(np.maximum(
            center[bits] - lower[bits], upper[bits] - center[bits]))),
        "legal_bits_sha256": hashlib.sha256(bits.astype("<u2").tobytes()).hexdigest(),
        "lower_table_sha256": hashlib.sha256(lower[bits].astype("<f8").tobytes()).hexdigest(),
        "upper_table_sha256": hashlib.sha256(upper[bits].astype("<f8").tobytes()).hexdigest(),
        "oracle_dependency_hashes": dependency_hashes(),
        "method": "Direct mathematical MPFR expm1 with RoundDown and RoundUp, then outward conversion of each endpoint to binary64. Independent mpmath expm1 must lie inside every MPFR interval.",
        "independence": "The oracle uses no candidate exponent reduction, split log(2), polynomial coefficients, or FP32 reconstruction.",
        "scope": "Exhaustive scalar-domain oracle. Tensor indexing and GPU execution are checked separately.",
    }
    for array in (lower, upper, center):
        array.setflags(write=False)
    return lower, upper, center, metadata


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all() and (np.abs(array) <= 10).all()
    return reference_table()[2][array.view(np.uint16)]


def compare(observed, bits):
    """Bound absolute errors and the mixed tolerance independently outward."""
    lower, upper, center, _ = reference_table()
    y = np.asarray(observed, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint16)
    assert y.shape == bits.shape and y.ndim == 1 and y.size > 0
    lo, hi, ref = lower[bits], upper[bits], center[bits]
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    finite = np.isfinite(y)
    far = np.nextafter(np.maximum(np.abs(y - lo), np.abs(y - hi)), np.inf)
    near = np.maximum(np.nextafter(np.maximum(np.maximum(lo - y, y - hi), 0.0), -np.inf), 0.0)
    smallest = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    largest = np.maximum(np.abs(lo), np.abs(hi))
    tolerance_lo = np.nextafter(ATOL + np.nextafter(RTOL * smallest, -np.inf), -np.inf)
    tolerance_hi = np.nextafter(ATOL + np.nextafter(RTOL * largest, np.inf), np.inf)
    worst = np.nextafter(far / tolerance_lo, np.inf)
    best = np.maximum(np.nextafter(near / tolerance_hi, -np.inf), 0.0)
    failing = (~finite) | (best > 1.0)
    ambiguous = finite & (best <= 1.0) & (worst > 1.0)
    index = int(np.argmax(np.where(finite, worst, np.inf)))
    witness = None
    if failing.any() or ambiguous.any():
        witness = {"index": index, "input_bits_hex": f"0x{int(bits[index]):04x}",
                   "input": float(bits[index:index + 1].view(np.float16)[0]),
                   "actual": float(y[index]), "reference": float(ref[index]),
                   "reference_lower": float(lo[index]), "reference_upper": float(hi[index]),
                   "abs_error_lower_bound": float(near[index]),
                   "abs_error_upper_bound": float(far[index]),
                   "allowed_error_lower_bound": float(tolerance_lo[index]),
                   "allowed_error_upper_bound": float(tolerance_hi[index]),
                   "error_ratio_lower_bound": float(best[index]),
                   "error_ratio_upper_bound": float(worst[index])}
    return {"numeric_passed": bool(finite.all() and not failing.any() and not ambiguous.any()),
            "finite": bool(finite.all()), "elements": int(y.size),
            "definite_failures": int(failing.sum()), "interval_ambiguous_elements": int(ambiguous.sum()),
            "max_abs_error": float(np.max(np.abs(y - ref))),
            "max_abs_error_upper_bound": float(np.max(far)),
            "max_error_ratio": float(np.max(worst)),
            "minimum_tolerance_margin": float(np.min(tolerance_lo - far)),
            "worst_input_bits_hex": f"0x{int(bits[index]):04x}",
            "worst_input": float(bits[index:index + 1].view(np.float16)[0]), "witness": witness}


def plans(seed=97000):
    lengths = (1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257,
               511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097,
               37377, 37379, 65535, 65536, 65537, 131071, 131072)
    rows = [dict(n=n, style="random_bits", seed=seed + i, offset=0)
            for i, n in enumerate(lengths)]
    for i, style in enumerate(("all_legal", "all_legal_reverse", "all_legal_shuffle")):
        rows.append(dict(n=37378, style=style, seed=seed + 100 + i, offset=0))
    rows.extend([
        dict(n=37378, style="all_legal_shuffle", seed=seed + 110, offset=7),
        dict(n=257, style="boundary_bits", seed=seed + 111, offset=1),
        dict(n=131072, style="boundary_bits", seed=seed + 112, offset=13),
        dict(n=2046, style="subnormals", seed=seed + 113, offset=3),
        dict(n=129, style="signed_zeros", seed=seed + 114, offset=5),
    ])
    return rows


def arrays(plan):
    bits = legal_bits()
    rng = np.random.default_rng(plan["seed"])
    style = plan["style"]
    if style.startswith("all_legal"):
        bits = bits.copy()
        if style.endswith("reverse"):
            bits = bits[::-1].copy()
        elif style.endswith("shuffle"):
            rng.shuffle(bits)
    elif style == "boundary_bits":
        positive = [0, 1, 2, 0x03FF, 0x0400, 0x0401, 0x2FFF, 0x3000, 0x3001,
                    0x37FF, 0x3800, 0x3801, 0x3BFF, 0x3C00, 0x3C01, 0x3FFF,
                    0x4000, 0x4001, 0x43FF, 0x4400, 0x4401, 0x48FF, 0x4900]
        boundary = np.array(positive + [b | 0x8000 for b in positive],
                            dtype=np.uint16)
        bits = np.resize(boundary, plan["n"])
    elif style == "subnormals":
        bits = bits[((bits & 0x7C00) == 0) & ((bits & 0x3FF) != 0)]
    elif style == "signed_zeros":
        bits = np.resize(np.array([0, 0x8000], dtype=np.uint16), plan["n"])
    else:
        bits = rng.choice(bits, size=plan["n"], replace=True)
    assert bits.size == plan["n"] and bits.dtype == np.uint16
    assert np.isfinite(bits.view(np.float16)).all() and (np.abs(bits.view(np.float16)) <= 10).all()
    return bits.view(np.float16).copy()


def cpu_audit(case_root):
    table = reference_table()
    values = legal_bits().view(np.float16)
    checks = []
    for name in EXPECTED:
        public = core.public_cpu(case_root, name)
        tensor = torch.from_numpy(values.copy())
        public["validate_inputs"](tensor, require_cuda=False)
        result = core.public_reference_check(
            public["reference"](tensor).detach().cpu().numpy(),
            values.view(np.uint16), table, ATOL, REFERENCE_BUDGET)
        invalid_inputs = (
            torch.tensor([-10.0078125], dtype=torch.float16),
            torch.tensor([10.0078125], dtype=torch.float16),
            torch.tensor([float("nan")], dtype=torch.float16),
            torch.tensor([float("inf")], dtype=torch.float16),
            torch.empty(0, dtype=torch.float16),
            torch.zeros(131073, dtype=torch.float16),
            torch.zeros(1, dtype=torch.float32),
        )
        for invalid in invalid_inputs:
            try:
                public["validate_inputs"](invalid, require_cuda=False)
            except ValueError:
                pass
            else:
                raise AssertionError((name, "accepted input outside the domain"))
        checks.append({"case": name, **result, "invalid_input_checks": len(invalid_inputs)})
    return {"passed": True, "oracle": table[3], "indexing": core.indexing_audit(),
            "public_references": checks, "oracle_dependency_hashes": dependency_hashes(),
            "scope": "Full legal-FP16 scalar reference audit; GPU arithmetic is checked separately."}


def validate(case_root):
    cpu = cpu_audit(case_root)
    table = reference_table()
    cases = {}
    for name, label in EXPECTED.items():
        module = core._load(case_root, name, "expm1")
        records = []
        for plan in plans():
            values, offset = arrays(plan), int(plan["offset"])
            backing = torch.full((offset + values.size + 23,), 0.3125,
                                 dtype=torch.float16, device="cuda")
            x = backing[offset:offset + values.size]
            x.copy_(torch.from_numpy(values).cuda())
            assert x.is_contiguous() and x.storage_offset() == offset
            before = backing.clone()
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            valid = (type(actual) is torch.Tensor and actual.dtype == torch.float32
                     and actual.device == x.device and tuple(actual.shape) == tuple(x.shape))
            assert valid, (name, plan, "output type/dtype/device/shape")
            numeric = compare(actual.detach().cpu().numpy(), values.view(np.uint16))
            preserved = bool(torch.equal(before.view(torch.uint8), backing.view(torch.uint8)))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            public = core.public_reference_check(
                module.reference(x).detach().cpu().numpy(),
                values.view(np.uint16), table, ATOL, REFERENCE_BUDGET)
            records.append(dict(
                plan, legal=True, **numeric,
                passed=bool(numeric["numeric_passed"] and preserved and fresh),
                inputs_and_backing_guards_preserved=preserved, inputs_preserved=preserved,
                output_fresh=bool(fresh), shape_dtype_valid=True,
                storage_offset=int(x.storage_offset()),
                public_reference_max_error_bound=public["max_abs_error_upper_bound"],
                unique_input_encodings=int(np.unique(values.view(np.uint16)).size),
                exhaustive_scalar_domain=plan["style"].startswith("all_legal")))
        failures = [row for row in records if not row["passed"]]
        numeric_reject = any(row["definite_failures"] > 0 for row in records)
        cases[name] = {
            "ground_truth": label, "tests": records, "tests_total": len(records),
            "tests_passed": len(records) - len(failures), "counterexamples": failures,
            "qualification_passed": numeric_reject if label == "reject" else not failures,
            "scalar_domain_encodings": 37378,
            "exhaustive_scalar_passes": sum(row["exhaustive_scalar_domain"] and row["passed"]
                                           for row in records),
            "gpu_elements_compared": sum(row["elements"] for row in records),
            "oracle_dependency_hashes": dependency_hashes(),
            "label_scope": "One fixed implementation over every legal input; input changes only supply witnesses.",
            "analysis": "Every legal FP16 encoding is checked against a directed mathematical interval in several orders and offsets. Separate tests cover lengths, tails, guards, input preservation and fresh output. This is not exhaustive testing of all tensor addresses or GPU environments.",
        }
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": cpu["passed"] and all(
                row["qualification_passed"] for row in cases.values())}
