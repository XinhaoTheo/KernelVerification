"""Independent pseudo-Huber qualification for fixed case_65/139.

Mixed absolute/relative tolerance is handled here without altering older
families or the shared absolute-only scalar oracle. The mathematical oracle
uses the direct real sqrt expression; public FP64 and rationalized candidate
use an algebraically different expression.
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

EXPECTED = {"case_65": "trust", "case_66": "reject"}
ATOL = 1e-8
RTOL = 1e-5
REFERENCE_BUDGET = 1e-10


def dependency_hashes():
    return {**core.dependency_hashes(), Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


@lru_cache(maxsize=1)
def reference_table():
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    bounds = {}
    for bit in range(0x7C00):
        value = float(np.array([bit], dtype=np.uint16).view(np.float16)[0])
        with down:
            x = gmpy2.mpfr(value)
            low = gmpy2.sqrt(1 + x * x) - 1
        with up:
            x = gmpy2.mpfr(value)
            high = gmpy2.sqrt(1 + x * x) - 1
        assert 0 <= low <= high
        bounds[bit] = bounds[bit | 0x8000] = (low, high)

    def independent_target(x):
        return (x * x) / (mp.sqrt(1 + x * x) + 1)

    return core.build_reference_table(bounds, independent_target, "mathematical_pseudo_huber", {
        "method": "Outward 320-bit MPFR multiplication, addition, sqrt and subtraction directly enclose sqrt(1+x*x)-1. Each stored FP16 value is exact. Independent 110-decimal mpmath crosschecks use x*x/(sqrt(1+x*x)+1).",
        "reference_expression_independence": "The directed mathematical oracle computes sqrt(1+x*x)-1 in high precision; public FP64 reference computes the rationalized equivalent. The oracle does not use candidate FP32 intermediates or GPU sqrt.",
        "mixed_tolerance": {"atol": ATOL, "rtol": RTOL},
        "oracle_dependency_hashes": dependency_hashes(),
    })


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all()
    return reference_table()[2][array.view(np.uint16)]


def compare(observed, bits, table=None):
    """Certify error versus every reference value in its outward interval.

    Since rho>=0, the allowed tolerance lies between atol+rtol*lo and
    atol+rtol*hi. Endpoint products/sums and distances are rounded outward
    in binary64. Pass uses upper error/lower tolerance; definite failure
    uses lower error/upper tolerance. Remaining overlap is not accepted.
    """
    lower, upper, center, _ = reference_table() if table is None else table
    y = np.asarray(observed, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint16)
    assert y.shape == bits.shape and y.ndim == 1 and y.size > 0
    lo, hi, ref = lower[bits], upper[bits], center[bits]
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(lo >= 0)
    finite = np.isfinite(y)
    worst = np.nextafter(np.maximum(np.abs(y - lo), np.abs(y - hi)), np.inf)
    best = np.maximum(np.nextafter(np.maximum(np.maximum(lo - y, y - hi), 0.0), -np.inf), 0.0)
    # Enclose the decimal constants themselves, not just FP64 operation rounding.
    atol_low, atol_high = np.nextafter(ATOL, -np.inf), np.nextafter(ATOL, np.inf)
    rtol_low, rtol_high = np.nextafter(RTOL, -np.inf), np.nextafter(RTOL, np.inf)
    allowed_low = np.nextafter(atol_low + np.nextafter(rtol_low * lo, -np.inf), -np.inf)
    allowed_high = np.nextafter(atol_high + np.nextafter(rtol_high * hi, np.inf), np.inf)
    failing = (~finite) | (best > allowed_high)
    uncertain = finite & (best <= allowed_high) & (worst > allowed_low)
    ratios = worst / allowed_low
    index = int(np.argmax(np.where(finite, ratios, np.inf)))
    witness = None
    if failing.any() or uncertain.any():
        witness = {"index": index, "input_bits_hex": f"0x{int(bits[index]):04x}",
                   "input": float(bits[index:index + 1].view(np.float16)[0]),
                   "actual": float(y[index]), "reference": float(ref[index]),
                   "reference_lower": float(lo[index]), "reference_upper": float(hi[index]),
                   "absolute_error_lower_bound": float(best[index]),
                   "absolute_error_upper_bound": float(worst[index]),
                   "allowed_error_lower_bound": float(allowed_low[index]),
                   "allowed_error_upper_bound": float(allowed_high[index]),
                   "allowed_error": float(ATOL + RTOL * abs(ref[index]))}
    return {"numeric_passed": bool(finite.all() and not failing.any() and not uncertain.any()),
            "finite": bool(finite.all()), "elements": int(y.size),
            "definite_failures": int(failing.sum()), "interval_ambiguous_elements": int(uncertain.sum()),
            "max_abs_error": float(np.max(np.abs(y - ref))),
            "max_abs_error_upper_bound": float(np.max(worst)),
            "max_error_ratio": float(np.max(ratios)),
            "minimum_tolerance_margin": float(np.min(allowed_low - worst)),
            "worst_input_bits_hex": f"0x{int(bits[index]):04x}",
            "worst_input": float(bits[index:index + 1].view(np.float16)[0]),
            "atol": ATOL, "rtol": RTOL, "witness": witness}


def plans():
    return core.plans(seed=138000)


def arrays(plan):
    return core.arrays(plan)


def cpu_audit(case_root):
    table = reference_table()
    values = core.finite_bit_patterns().view(np.float16)
    checks = []
    for name in EXPECTED:
        public = core.public_cpu(case_root, name)
        tensor = torch.from_numpy(values.copy())
        public["validate_inputs"](tensor, require_cuda=False)
        observed = public["reference"](tensor).detach().cpu().numpy()
        result = compare(observed, values.view(np.uint16), table)
        assert result["numeric_passed"] and result["max_abs_error_upper_bound"] <= REFERENCE_BUDGET, result
        checks.append({"case": name, **result, "public_reference_dtype": str(observed.dtype),
                       "absolute_reference_budget": REFERENCE_BUDGET})
    return {"passed": True, "oracle": table[3], "indexing": core.indexing_audit(),
            "public_references": checks, "oracle_dependency_hashes": dependency_hashes(),
            "scope": "Full finite-FP16 scalar reference audit, distinct from GPU arithmetic and tensor-property qualification."}


def cpu_screen():
    """Two preregistered CPU models; not GPU evidence or model-agent trials."""
    bits = core.finite_bit_patterns()
    x = bits.view(np.float16).astype(np.float32)
    square = np.multiply(x, x, dtype=np.float32)
    root = np.sqrt(np.add(np.float32(1), square, dtype=np.float32), dtype=np.float32)
    candidates = [
        ("case_65", "rationalized", np.divide(square, np.add(root, np.float32(1), dtype=np.float32), dtype=np.float32)),
        ("case_66", "direct_subtraction", np.subtract(root, np.float32(1), dtype=np.float32)),
    ]
    return [{"case": name, "expression": expression, "evaluation": "CPU NumPy FP32 separate operations; GPU sqrt may differ", **compare(result, bits)}
            for name, expression, result in candidates]


def validate(case_root):
    cpu = cpu_audit(case_root)
    table = reference_table()
    cases = {}
    for name, label in EXPECTED.items():
        module = core._load(case_root, name, "pseudo_huber")
        records = []
        for plan in plans():
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
            numeric = compare(actual.detach().cpu().numpy(), values.view(np.uint16), table)
            preserved = bool(torch.equal(before.view(torch.uint8), backing.view(torch.uint8)))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            public = compare(module.reference(x).detach().cpu().numpy(), values.view(np.uint16), table)
            assert public["numeric_passed"] and public["max_abs_error_upper_bound"] <= REFERENCE_BUDGET, public
            records.append(dict(plan, legal=True, **numeric, passed=bool(numeric["numeric_passed"] and preserved and fresh),
                                inputs_and_backing_guards_preserved=preserved, inputs_preserved=preserved,
                                output_fresh=bool(fresh), shape_dtype_valid=True, storage_offset=int(x.storage_offset()),
                                public_reference_max_error_bound=public["max_abs_error_upper_bound"],
                                public_reference_interval_certified_fp32=False,
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
                       "atol": ATOL, "rtol": RTOL,
                       "label_scope": "One fixed implementation over the full public domain; changing inputs supplies witnesses, never labels.",
                       "analysis": "Every finite FP16 encoding compared with an outward mathematical interval and mixed tolerance in several orders and alignments. Lengths/tails, backing guards, input preservation and fresh storage checked separately. Scalar coverage is not every GPU specialization or memory execution."}
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": all(row["qualification_passed"] for row in cases.values())}
