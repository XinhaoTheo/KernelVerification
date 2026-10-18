"""Directed, correctly-rounded binary16 sigmoid oracle for case_148/149.

Every finite input encoding is covered. MPFR directed exp/reciprocal bounds
are rounded to binary16 by integer arithmetic, without a binary32/binary64
intermediate conversion. Independent mpmath checks use tanh(x/2).
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib
import struct
from pathlib import Path

import gmpy2
import mpmath as mp
import numpy as np
import torch

try:
    from . import adaptive_unary_oracle as core
except ImportError:
    import adaptive_unary_oracle as core

EXPECTED = {"case_148": "trust", "case_149": "reject"}
TAIL_CUTOFF = 32


def dependency_hashes():
    return {**core.dependency_hashes(), Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _nearest_even_integer(numerator, denominator):
    q, r = divmod(int(numerator), int(denominator))
    return q + int(2 * r > denominator or (2 * r == denominator and q % 2 == 1))


def round_positive_ratio_to_half(numerator, denominator):
    """Exact RNE binary16 encoding of a rational in [0,1], using integers only."""
    numerator, denominator = int(numerator), int(denominator)
    assert 0 <= numerator <= denominator and denominator > 0
    if numerator == 0:
        return 0
    if numerator * (1 << 14) < denominator:
        # All binary16 subnormals have the same quantum 2^-24. A rounded
        # significand of 1024 naturally becomes the smallest normal.
        return _nearest_even_integer(numerator << 24, denominator)
    exponent = numerator.bit_length() - denominator.bit_length()
    if (numerator << max(-exponent, 0)) < (denominator << max(exponent, 0)):
        exponent -= 1
    assert -14 <= exponent <= 0
    significand = _nearest_even_integer(numerator << (10 - exponent), denominator)
    # Carry at significand=2048 increments the encoded exponent directly.
    return ((exponent + 15) << 10) + significand - 1024


@lru_cache(maxsize=1)
def _rounder_audit():
    """Independent exact-dyadic round trips and every midpoint from zero to one."""
    values = np.arange(0x3C01, dtype=np.uint16).view(np.float16)
    dyadics = [Fraction.from_float(float(x)) for x in values]
    for bit, value in enumerate(dyadics):
        assert round_positive_ratio_to_half(value.numerator, value.denominator) == bit
    for bit, (low, high) in enumerate(zip(dyadics, dyadics[1:])):
        midpoint = (low + high) / 2
        expected = bit if bit % 2 == 0 else bit + 1
        assert round_positive_ratio_to_half(midpoint.numerator, midpoint.denominator) == expected
        for point, wanted in [((low + midpoint) / 2, bit), ((midpoint + high) / 2, bit + 1)]:
            assert round_positive_ratio_to_half(point.numerator, point.denominator) == wanted
    return {"passed": True, "exact_representable_round_trips": len(dyadics),
            "adjacent_midpoint_ties_checked": len(dyadics) - 1,
            "quarter_interval_checks": 2 * (len(dyadics) - 1),
            "method": "Python Fraction exact dyadics, parity-based tie expectation, and quarter-interval points independently validate integer rounding."}


@lru_cache(maxsize=1)
def reference_table():
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    with up:
        tail_upper = gmpy2.exp(gmpy2.mpfr(-TAIL_CUTOFF))
    with down:
        positive_tail_lower = 1 - tail_upper
    bounds = {}
    expected = np.full(65536, 0xFFFF, dtype=np.uint16)
    tails = 0
    for bit in core.finite_bit_patterns():
        value = float(np.array([bit], dtype=np.uint16).view(np.float16)[0])
        if value >= TAIL_CUTOFF:
            low, high = positive_tail_lower, gmpy2.mpfr(1)
            tails += 1
        elif value <= -TAIL_CUTOFF:
            low, high = gmpy2.mpfr(0), tail_upper
            tails += 1
        else:
            with down:
                exponential_low = gmpy2.exp(-gmpy2.mpfr(value))
                denominator_low = 1 + exponential_low
            with up:
                exponential_high = gmpy2.exp(-gmpy2.mpfr(value))
                denominator_high = 1 + exponential_high
            with down:
                low = 1 / denominator_high
            with up:
                high = 1 / denominator_low
        assert 0 <= low <= high <= 1
        low_bit = round_positive_ratio_to_half(*low.as_integer_ratio())
        high_bit = round_positive_ratio_to_half(*high.as_integer_ratio())
        assert low_bit == high_bit, (hex(int(bit)), str(low), str(high))
        bounds[int(bit)] = (low, high)
        expected[bit] = low_bit

    def independent_target(x):
        return (1 + mp.tanh(x / 2)) / 2

    table = core.build_reference_table(bounds, independent_target, "correctly_rounded_fp16_sigmoid", {
        "method": "MPFR 320-bit directed exp(-x), addition and reciprocal. Lower sigmoid bound uses an upward denominator; upper bound uses a downward denominator. Exact integer-ratio RNE rounding of both endpoints must select the identical binary16 encoding.",
        "independent_crosscheck": "110-digit mpmath uses (1+tanh(x/2))/2, algebraically different from the oracle and candidate stable exp/division paths.",
        "tail_cutoff": TAIL_CUTOFF, "tail_bound_encodings": tails,
        "tail_argument": "For x<=-32, 0<sigmoid(x)<exp(-32); for x>=32, 1-exp(-32)<sigmoid(x)<1. The directed bounds both round to +0 or 1 respectively; no input is removed from the domain.",
        "tail_upper_bound": str(tail_upper),
        "rounding_mode": "binary16 round-to-nearest, ties-to-even",
        "rounding_interval_certified_encodings": int(core.finite_bit_patterns().size),
        "integer_rounder_audit": _rounder_audit(),
        "expected_fp16_table_sha256": hashlib.sha256(expected[core.finite_bit_patterns()].astype('<u2').tobytes()).hexdigest(),
        "oracle_dependency_hashes": dependency_hashes(),
    })
    expected.setflags(write=False)
    return (*table, expected)


def independent_reference(values):
    values = np.asarray(values)
    assert values.dtype == np.float16 and np.isfinite(values).all()
    return reference_table()[4][values.view(np.uint16)].view(np.float16)


def compare(observed, bits, table=None):
    table = reference_table() if table is None else table
    observed = np.asarray(observed)
    bits = np.asarray(bits, dtype=np.uint16)
    assert observed.dtype == np.float16 and observed.ndim == 1 and observed.shape == bits.shape and bits.size
    actual = observed.view(np.uint16)
    expected = table[4][bits]
    failures = actual != expected
    examples = []
    for i in np.flatnonzero(failures)[:8]:
        examples.append({"index": int(i), "input_bits_hex": f"0x{int(bits[i]):04x}",
                         "input": float(bits[i:i+1].view(np.float16)[0]),
                         "actual_bits_hex": f"0x{int(actual[i]):04x}", "actual": float(observed[i]),
                         "expected_bits_hex": f"0x{int(expected[i]):04x}",
                         "expected": float(expected[i:i+1].view(np.float16)[0]),
                         "mathematical_reference_lower": float(table[0][bits[i]]),
                         "mathematical_reference_upper": float(table[1][bits[i]])})
    finite = bool(np.isfinite(observed).all())
    return {"numeric_passed": bool(not failures.any()), "finite": finite,
            "elements": int(bits.size), "definite_failures": int(failures.sum()),
            "interval_ambiguous_elements": 0,
            "maximum_fp16_encoding_distance": int(np.max(np.abs(actual.astype(np.int32) - expected.astype(np.int32)))),
            "comparison": "exact FP16 bits versus directed-interval-certified RNE result",
            "witness": examples[0] if examples else None, "first_mismatches": examples}


def plans():
    return core.plans(seed=148000)


def arrays(plan):
    return core.arrays(plan)


def cpu_screen():
    bits = core.finite_bit_patterns()
    results = []
    for name, dtype in [("case_148", np.float64), ("case_149", np.float32)]:
        x = bits.view(np.float16).astype(dtype)
        z = np.exp(-np.abs(x))
        denominator = dtype(1) + z
        y = np.where(x < 0, z / denominator, dtype(1) / denominator).astype(np.float16)
        results.append({"case": name, "candidate": f"stable_sigmoid_{dtype.__name__}_then_fp16",
                        "method": "CPU NumPy exp/division, distinct from libdevice; only screening, not GPU qualification",
                        **compare(y, bits)})
    return results


def cpu_audit(case_root):
    table = reference_table()
    values = core.finite_bit_patterns().view(np.float16)
    checks = []
    for name in EXPECTED:
        public = core.public_cpu(case_root, name)
        x = torch.from_numpy(values.copy())
        public['validate_inputs'](x, require_cuda=False)
        actual = public['reference'](x).detach().cpu().numpy()
        result = compare(actual, values.view(np.uint16), table)
        assert result['numeric_passed'], result
        checks.append({"case": name, **result, "public_reference_dtype": str(actual.dtype)})
    # Exercise CPython's direct float64-to-binary16 packing on every exact
    # representable value and every adjacent midpoint in sigmoid's [0,1] range.
    values01 = np.arange(0x3C01, dtype=np.uint16).view(np.float16).astype(np.float64)
    midpoints = (values01[:-1] + values01[1:]) / 2
    packed_values = np.frombuffer(b"".join(struct.pack("<e", float(v)) for v in values01), dtype='<u2')
    packed_midpoints = np.frombuffer(b"".join(struct.pack("<e", float(v)) for v in midpoints), dtype='<u2')
    expected_values = np.arange(0x3C01, dtype=np.uint16)
    expected_midpoints = np.arange(0x3C00, dtype=np.uint16)
    expected_midpoints += expected_midpoints & 1
    assert np.array_equal(packed_values, expected_values)
    assert np.array_equal(packed_midpoints, expected_midpoints)
    conversion = {"primitive": "CPython struct.pack('<e', Python float64)",
                  "passed": True, "exact_output_values_checked": int(values01.size),
                  "adjacent_midpoint_ties_checked": int(midpoints.size),
                  "reason": "Explicit half packing avoids a platform-specific Torch float64-to-half path that may pass through FP32. Whole sigmoid references are also checked against the independent oracle above."}
    return {"passed": True, "oracle": table[3], "indexing": core.indexing_audit(),
            "public_reference_conversion": conversion,
            "public_references": checks, "oracle_dependency_hashes": dependency_hashes(),
            "scope": "Every finite scalar input and both public CPU reference functions checked; GPU libdevice arithmetic/conversions require separate T4 validation."}


def validate(case_root):
    cpu = cpu_audit(case_root)
    table = reference_table()
    cases = {}
    for name, label in EXPECTED.items():
        module = core._load(case_root, name, 'sigmoid_rounding')
        records = []
        for plan in plans():
            values, offset = arrays(plan), int(plan['offset'])
            backing = torch.full((offset + values.size + 23,), 0.3125, dtype=torch.float16, device='cuda')
            x = backing[offset:offset + values.size]
            x.copy_(torch.from_numpy(values).cuda())
            assert x.is_contiguous() and x.storage_offset() == offset
            before = backing.clone()
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            valid = type(actual) is torch.Tensor and actual.dtype == torch.float16 and actual.device == x.device and tuple(actual.shape) == tuple(x.shape)
            assert valid, (name, plan, 'output type/dtype/device/shape')
            numeric = compare(actual.detach().cpu().numpy(), values.view(np.uint16), table)
            preserved = bool(torch.equal(before.view(torch.uint8), backing.view(torch.uint8)))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            public = compare(module.reference(x).detach().cpu().numpy(), values.view(np.uint16), table)
            assert public['numeric_passed'], public
            records.append(dict(plan, legal=True, **numeric,
                                passed=bool(numeric['numeric_passed'] and preserved and fresh),
                                inputs_and_backing_guards_preserved=preserved, inputs_preserved=preserved,
                                output_fresh=bool(fresh), shape_dtype_valid=True,
                                storage_offset=int(x.storage_offset()), public_reference_exact=True,
                                unique_input_encodings=int(np.unique(values.view(np.uint16)).size),
                                exhaustive_scalar_domain=plan['style'].startswith('all_finite')))
        failures = [row for row in records if not row['passed']]
        numeric_reject = any(row['definite_failures'] > 0 for row in records)
        cases[name] = {"ground_truth": label, "tests": records,
                       "tests_total": len(records), "tests_passed": len(records)-len(failures),
                       "counterexamples": failures,
                       "qualification_passed": numeric_reject if label == 'reject' else not failures and cpu['passed'],
                       "scalar_domain_encodings": 63488,
                       "exhaustive_scalar_passes": sum(row['exhaustive_scalar_domain'] and row['passed'] for row in records),
                       "gpu_elements_compared": sum(row['elements'] for row in records),
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One fixed implementation label across the full public domain, with exact final FP16 rounding.",
                       "analysis": "Every finite FP16 encoding is compared with an independently certified binary16 rounding result in multiple orders/offsets. Lengths, masked tails, signed zeros, subnormals, input/backing preservation and output freshness are tested separately; scalar exhaustion does not cover every tensor execution."}
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": all(row['qualification_passed'] for row in cases.values())}
