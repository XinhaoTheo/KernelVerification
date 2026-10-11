"""Directed mathematical SiLU qualification for the fixed case_130/131 pair.

The numerical target uses exp directly and is independent of the candidates'
base-two exponential argument reduction and serialized constants.
"""
from __future__ import annotations

from functools import lru_cache

import gmpy2
import mpmath as mp
import numpy as np

try:
    from . import adaptive_unary_oracle as core
except ImportError:
    import adaptive_unary_oracle as core

EXPECTED = {"case_130": "trust", "case_131": "reject"}
ATOL = 1e-6
REFERENCE_BUDGET = 1e-10
TAIL_CUTOFF = 32


@lru_cache(maxsize=1)
def reference_table():
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    with up:
        tail_upper = TAIL_CUTOFF * gmpy2.exp(gmpy2.mpfr(-TAIL_CUTOFF))
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
                exponential_low = gmpy2.exp(-a)
                denominator_low = 1 + exponential_low
            with up:
                exponential_high = gmpy2.exp(-a)
                denominator_high = 1 + exponential_high
            # Positive interval division: combine the numerator's lower bound
            # with the denominator's upper bound (and conversely). The shared
            # exponential creates correlation, but losing it widens safely.
            with down:
                delta_low = (a * exponential_low) / denominator_high
            with up:
                delta_high = (a * exponential_high) / denominator_low
        with down:
            positive_low = a - delta_high
            negative_low = -delta_high
        with up:
            positive_high = a - delta_low
            negative_high = -delta_low
        bounds[bit] = (positive_low, positive_high)
        bounds[bit | 0x8000] = (negative_low, negative_high)

    def target(x):
        # Independent expression keeps every exponential argument nonpositive.
        exponential = mp.exp(-abs(x))
        return x / (1 + exponential) if x >= 0 else x * exponential / (1 + exponential)

    return core.build_reference_table(bounds, target, "mathematical_silu", {
        "method": "For a=|x|, enclose delta=a*exp(-a)/(1+exp(-a)) using directed exp, positive multiplication, addition and division. SiLU(a)=a-delta and SiLU(-a)=-delta; each subtraction and negation is directed outward.",
        "tail_cutoff": TAIL_CUTOFF,
        "tail_bound_encodings": 2 * tail_magnitudes,
        "tail_correction_upper_bound": str(tail_upper),
        "tail_bound_argument": "For a>=32, 0<=delta(a)<=a*exp(-a)<=32*exp(-32), since the derivative of a*exp(-a) is (1-a)*exp(-a)<=0 for a>=1. The upper bound is computed with upward rounding. This avoids huge negative exponents and large integer-ratio allocations for extreme FP16 values.",
        "coefficient_independence": "The oracle does not use the candidates' exp2 implementation, LOG2E constant, or arithmetic helpers.",
    })


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all()
    return reference_table()[2][array.view(np.uint16)]


def plans():
    return core.plans(seed=130000)


def arrays(plan):
    return core.arrays(plan)


def cpu_audit(case_root):
    return core.cpu_audit(case_root, EXPECTED, reference_table(), ATOL, REFERENCE_BUDGET)


def validate(case_root):
    return core.validate(case_root, EXPECTED, reference_table(), "silu", ATOL, REFERENCE_BUDGET, seed=130000)
