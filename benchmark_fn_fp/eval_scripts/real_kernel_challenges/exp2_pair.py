"""Independent qualification of BF16 exp2 kernels, with directed MPFR bounds.

The oracle calls exp2 directly. It does not use the candidates' range reduction,
polynomial, rounded coefficients or FP32 evaluation.
"""
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

EXPECTED = {"case_91": "reject", "case_92": "trust"}
RTOL = 1e-6
REFERENCE_RELATIVE_BUDGET = 1e-12
ORACLE_BITS = 320
MPMATH_DIGITS = 110


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest(),
            **core.dependency_hashes()}


def decode(bits):
    """Decode exact BF16 encodings into exactly representable FP32 values."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def legal_bits():
    bits = np.arange(65536, dtype=np.uint16)
    values = decode(bits)
    selected = bits[np.isfinite(values) & (np.abs(values) <= 10)]
    assert selected.size == 33346
    return selected


def tensor_from_bits(bits):
    return torch.from_numpy(np.asarray(bits, dtype=np.uint16).copy()).view(torch.bfloat16)


@lru_cache(maxsize=1)
def reference_table():
    bits = legal_bits()
    lower = np.full(65536, np.nan, dtype=np.float64)
    upper, center = lower.copy(), lower.copy()
    down = gmpy2.context(precision=ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=ORACLE_BITS, round=gmpy2.RoundUp)
    down64 = gmpy2.context(precision=53, round=gmpy2.RoundDown)
    up64 = gmpy2.context(precision=53, round=gmpy2.RoundUp)
    exact64 = gmpy2.context(precision=ORACLE_BITS, round=gmpy2.RoundToNearest)
    with mp.workdps(MPMATH_DIGITS):
        for bit, value in zip(bits, decode(bits)):
            value = float(value)
            with down:
                lo = gmpy2.exp2(gmpy2.mpfr(value))
            with up:
                hi = gmpy2.exp2(gmpy2.mpfr(value))
            assert 0 < lo <= hi and gmpy2.is_finite(hi)
            other = mp.power(2, mp.mpf(value))
            ln, ld = lo.as_integer_ratio()
            hn, hd = hi.as_integer_ratio()
            assert mp.mpf(int(ln)) / int(ld) <= other <= mp.mpf(int(hn)) / int(hd)
            with down64:
                lower[bit] = float(gmpy2.mpfr(lo, 53))
            with up64:
                upper[bit] = float(gmpy2.mpfr(hi, 53))
            with exact64:
                assert gmpy2.mpfr(float(lower[bit])) <= lo
                assert gmpy2.mpfr(float(upper[bit])) >= hi
            center[bit] = float(other)
    assert np.all(lower[bits] <= center[bits])
    assert np.all(center[bits] <= upper[bits])
    metadata = {
        "passed": True, "operation": "mathematical_exp2",
        "legal_bfloat16_encodings": int(bits.size),
        "excluded_encodings": 65536 - int(bits.size),
        "subnormal_encodings": int((((bits & 0x7F80) == 0) & ((bits & 0x7F) != 0)).sum()),
        "signed_zero_encodings": int(((bits & 0x7FFF) == 0).sum()),
        "mpfr_precision_bits": ORACLE_BITS, "mpmath_decimal_digits": MPMATH_DIGITS,
        "mpmath_independent_crosschecks": int(bits.size),
        "directed_binary64_endpoint_checks": 2 * int(bits.size),
        "gmpy2_version": gmpy2.version(), "mpfr_version": gmpy2.mpfr_version(),
        "mpmath_version": mp.__version__,
        "maximum_binary64_relative_interval_width": float(np.max((upper[bits] - lower[bits]) / lower[bits])),
        "legal_bits_sha256": hashlib.sha256(bits.astype("<u2").tobytes()).hexdigest(),
        "lower_table_sha256": hashlib.sha256(lower[bits].astype("<f8").tobytes()).hexdigest(),
        "upper_table_sha256": hashlib.sha256(upper[bits].astype("<f8").tobytes()).hexdigest(),
        "method": "Direct MPFR exp2 at 320 bits with RoundDown and RoundUp; binary64 endpoints are rounded outward and reconverted to check containment.",
        "independence": "The oracle does not use candidate range reduction, polynomial coefficients, bit-built scale factors or FP32 arithmetic.",
        "scope": "Exhaustive legal scalar encodings; tensor execution and indexing are checked separately.",
        "oracle_dependency_hashes": dependency_hashes(),
    }
    for array in (lower, upper, center):
        array.setflags(write=False)
    return lower, upper, center, metadata


def compare(observed, bits, relative_tolerance=RTOL):
    lower, upper, center, _ = reference_table()
    y = np.asarray(observed, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint16)
    assert y.shape == bits.shape and y.ndim == 1 and y.size > 0
    lo, hi, ref = lower[bits], upper[bits], center[bits]
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and (lo > 0).all()
    finite = np.isfinite(y)
    far = np.nextafter(np.maximum(np.abs(y - lo), np.abs(y - hi)), np.inf)
    near = np.maximum(np.nextafter(np.maximum(np.maximum(lo - y, y - hi), 0.0), -np.inf), 0.0)
    worst = np.nextafter(far / lo, np.inf)
    best = np.maximum(np.nextafter(near / hi, -np.inf), 0.0)
    failing = (~finite) | (best > relative_tolerance)
    ambiguous = finite & (best <= relative_tolerance) & (worst > relative_tolerance)
    index = int(np.argmax(np.where(finite, worst, np.inf)))
    witness = None
    if failing.any() or ambiguous.any():
        witness = {
            "index": index, "input_bits_hex": f"0x{int(bits[index]):04x}",
            "input": float(decode(bits[index:index + 1])[0]), "actual": float(y[index]),
            "reference": float(ref[index]), "reference_lower": float(lo[index]),
            "reference_upper": float(hi[index]),
            "relative_error_lower_bound": float(best[index]),
            "relative_error_upper_bound": float(worst[index]),
            "allowed_relative_error": relative_tolerance,
        }
    maximum = float(np.max(worst))
    return {
        "numeric_passed": bool(finite.all() and not failing.any() and not ambiguous.any()),
        "finite": bool(finite.all()), "elements": int(y.size),
        "definite_failures": int(failing.sum()), "interval_ambiguous_elements": int(ambiguous.sum()),
        "max_relative_error": float(np.max(np.abs(y - ref) / ref)),
        "max_relative_error_upper_bound": maximum,
        "max_error_ratio": maximum / relative_tolerance,
        "minimum_relative_tolerance_margin": relative_tolerance - maximum,
        "worst_input_bits_hex": f"0x{int(bits[index]):04x}",
        "worst_input": float(decode(bits[index:index + 1])[0]), "witness": witness,
    }


def plans():
    lengths = (1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513,
               1023, 1024, 1025, 4095, 4096, 4097, 33345, 33347, 65535, 65536,
               65537, 131071, 131072)
    result = [dict(n=n, style="random_bits", seed=163000 + i, offset=0) for i, n in enumerate(lengths)]
    result += [dict(n=33346, style=style, seed=163100 + i, offset=0)
               for i, style in enumerate(("all_legal", "all_legal_reverse", "all_legal_shuffle"))]
    result += [dict(n=33346, style="all_legal_shuffle", seed=163110, offset=7),
               dict(n=257, style="boundary_bits", seed=163111, offset=1),
               dict(n=131072, style="boundary_bits", seed=163112, offset=13),
               dict(n=254, style="subnormal_bits", seed=163113, offset=3),
               dict(n=129, style="signed_zeros", seed=163114, offset=5)]
    return result


def arrays(plan):
    allowed = legal_bits()
    rng = np.random.default_rng(plan["seed"])
    style = plan["style"]
    if style.startswith("all_legal"):
        bits = allowed.copy()
        if style.endswith("reverse"):
            bits = bits[::-1].copy()
        elif style.endswith("shuffle"):
            rng.shuffle(bits)
    elif style == "subnormal_bits":
        bits = allowed[((allowed & 0x7F80) == 0) & ((allowed & 0x7F) != 0)]
    elif style == "signed_zeros":
        bits = np.resize(np.array([0, 0x8000], dtype=np.uint16), plan["n"])
    elif style == "boundary_bits":
        positive = [0, 1, 0x007F, 0x0080, 0x0081, 0x3EFF, 0x3F00, 0x3F01,
                    0x3F7F, 0x3F80, 0x3F81, 0x3FBF, 0x3FC0, 0x3FC1,
                    0x401F, 0x4020, 0x4021, 0x410F, 0x4110, 0x4111, 0x411F, 0x4120]
        bits = np.resize(np.array(positive + [b | 0x8000 for b in positive], dtype=np.uint16), plan["n"])
    else:
        bits = rng.choice(allowed, size=plan["n"], replace=True)
    assert bits.size == plan["n"] and np.isin(bits, allowed).all()
    return bits


def cpu_audit(case_root):
    bits = legal_bits()
    records = []
    for name in EXPECTED:
        public = core.public_cpu(case_root, name)
        x = tensor_from_bits(bits)
        public["validate_inputs"](x, require_cuda=False)
        observed = public["reference"](x).detach().cpu().numpy()
        assert observed.dtype == np.float64
        result = compare(observed, bits, REFERENCE_RELATIVE_BUDGET)
        assert result["numeric_passed"], (name, result)
        records.append({"case": name, "public_reference_dtype": str(observed.dtype), **result})
    return {"passed": True, "oracle": reference_table()[3],
            "indexing": core.indexing_audit(), "public_references": records,
            "oracle_dependency_hashes": dependency_hashes(),
            "scope": "Exhaustive BF16 domain reference audit, separate from GPU arithmetic qualification."}


def validate(case_root):
    cpu = cpu_audit(case_root)
    cases = {}
    for name, label in EXPECTED.items():
        module = core._load(case_root, name, "exp2")
        records = []
        for plan in plans():
            bits, offset = arrays(plan), int(plan["offset"])
            backing = torch.full((offset + bits.size + 23,), 0.3125, dtype=torch.bfloat16, device="cuda")
            x = backing[offset:offset + bits.size]
            x.copy_(tensor_from_bits(bits).cuda())
            assert x.is_contiguous() and x.storage_offset() == offset
            before = backing.clone()
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            valid = type(actual) is torch.Tensor and actual.dtype == torch.float32 and actual.device == x.device and tuple(actual.shape) == tuple(x.shape)
            assert valid, (name, plan, "output type/dtype/device/shape")
            numeric = compare(actual.detach().cpu().numpy(), bits)
            preserved = bool(torch.equal(before.view(torch.uint8), backing.view(torch.uint8)))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            public = compare(module.reference(x).detach().cpu().numpy(), bits, REFERENCE_RELATIVE_BUDGET)
            assert public["numeric_passed"], (name, plan, "public reference")
            records.append(dict(
                plan, legal=True, **numeric, passed=bool(numeric["numeric_passed"] and preserved and fresh),
                inputs_and_backing_guards_preserved=preserved, inputs_preserved=preserved,
                output_fresh=bool(fresh), shape_dtype_valid=True, storage_offset=int(x.storage_offset()),
                public_reference_max_relative_error_bound=public["max_relative_error_upper_bound"],
                unique_input_encodings=int(np.unique(bits).size),
                exhaustive_scalar_domain=plan["style"].startswith("all_legal")))
        failures = [row for row in records if not row["passed"]]
        numeric_reject = any(row["definite_failures"] > 0 for row in records)
        cases[name] = {
            "ground_truth": label, "tests": records, "tests_total": len(records),
            "tests_passed": len(records) - len(failures), "counterexamples": failures,
            "qualification_passed": numeric_reject if label == "reject" else not failures and cpu["passed"],
            "scalar_domain_encodings": 33346,
            "exhaustive_scalar_passes": sum(row["exhaustive_scalar_domain"] and row["passed"] for row in records),
            "gpu_elements_compared": sum(row["elements"] for row in records),
            "oracle_dependency_hashes": dependency_hashes(),
            "label_scope": "One fixed implementation over the full public domain; inputs only supply witnesses.",
            "analysis": "All 33,346 legal BF16 encodings compared with directed mathematical intervals in multiple orders and offsets. Other N/tails, input guards and fresh output are checked separately. Scalar coverage is not exhaustive GPU execution coverage.",
        }
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": all(row["qualification_passed"] for row in cases.values())}
