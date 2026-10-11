"""Stable, directed-rounding mathematical GELU qualification for case_128/129."""
from __future__ import annotations

from functools import lru_cache

import gmpy2
import mpmath as mp
import numpy as np

try:
    from . import adaptive_unary_oracle as core
except ImportError:
    import adaptive_unary_oracle as core

EXPECTED = {"case_128": "trust", "case_129": "reject"}
ATOL = 1e-6
# Public reference returns FP64. The wider positive-tail interval contributes
# at most one binary64 ulp (7.28e-12 at the largest finite FP16 input).
REFERENCE_BUDGET = 1e-10
TAIL_CUTOFF = 16


@lru_cache(maxsize=1)
def reference_table():
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    with down:
        sqrt_two_low = gmpy2.sqrt(gmpy2.mpfr(2))
        density_denominator_low = gmpy2.sqrt(2 * gmpy2.const_pi())
    with up:
        sqrt_two_high = gmpy2.sqrt(gmpy2.mpfr(2))
        tail_upper = gmpy2.exp(gmpy2.mpfr(-128)) / density_denominator_low
    bounds = {}
    tail_magnitudes = 0
    for bit in range(0x7C00):
        value = float(np.array([bit], dtype=np.uint16).view(np.float16)[0])
        with down:
            a = gmpy2.mpfr(value)
        if value >= TAIL_CUTOFF:
            with down:
                delta_low = gmpy2.mpfr(0)
            delta_high = tail_upper
            tail_magnitudes += 1
        else:
            with down:
                z_low = a / sqrt_two_high
            with up:
                z_high = a / sqrt_two_low
            # erfc decreases, so its argument endpoints reverse.
            with down:
                delta_low = (a / 2) * gmpy2.erfc(z_high)
            with up:
                delta_high = (a / 2) * gmpy2.erfc(z_low)
        with down:
            positive_low = a - delta_high
            negative_low = -delta_high
        with up:
            positive_high = a - delta_low
            negative_high = -delta_low
        bounds[bit] = (positive_low, positive_high)
        bounds[bit | 0x8000] = (negative_low, negative_high)
    def target(x):
        return (x / 2) * mp.erfc(-x / mp.sqrt(2))
    return core.build_reference_table(bounds, target, "mathematical_gelu", {
        "method": "For a=|x|, enclose delta=(a/2)*erfc(a/sqrt(2)) using directed sqrt/division and erfc monotonicity. GELU(a)=a-delta and GELU(-a)=-delta; no 1+erf cancellation or incorrect odd-symmetry assumption.",
        "tail_cutoff": TAIL_CUTOFF, "tail_bound_encodings": 2 * tail_magnitudes,
        "tail_correction_upper_bound": str(tail_upper),
        "tail_bound_argument": "For u>0, integral_u^infinity exp(-t^2)dt <= integral_u^infinity (t/u)exp(-t^2)dt = exp(-u^2)/(2u). Thus delta(a)<=exp(-a^2/2)/sqrt(2*pi)<=exp(-128)/sqrt(2*pi) for a>=16. The bound is computed upward, then GELU intervals use outward subtraction. This avoids huge MPFR exponents and giant integer ratios for large negative inputs.",
        "coefficient_independence": "Neither candidate polynomial coefficients nor candidate arithmetic helpers participate in this oracle."})


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all()
    return reference_table()[2][array.view(np.uint16)]


def plans():
    return core.plans(seed=128000)


def arrays(plan):
    return core.arrays(plan)


def cpu_audit(case_root):
    return core.cpu_audit(case_root, EXPECTED, reference_table(), ATOL, REFERENCE_BUDGET)


def validate(case_root):
    return core.validate(case_root, EXPECTED, reference_table(), "gelu", ATOL, REFERENCE_BUDGET, seed=128000)
