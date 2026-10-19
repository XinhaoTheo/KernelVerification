"""Directed mathematical softplus qualification for case_63/137.

The private oracle evaluates log1p(exp(-abs(x))) without the candidates'
sqrt(2) range reduction or their polynomial approximation.
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

EXPECTED = {"case_63": "trust", "case_64": "reject"}
ATOL = 1e-6
REFERENCE_BUDGET = 1e-10
TAIL_CUTOFF = 32


@lru_cache(maxsize=1)
def reference_table():
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    with up:
        tail_upper = gmpy2.exp(gmpy2.mpfr(-TAIL_CUTOFF))
    bounds = {}
    tail_magnitudes = 0
    for bit in range(0x7C00):
        value = float(np.array([bit], dtype=np.uint16).view(np.float16)[0])
        with down:
            a = gmpy2.mpfr(value)
        if value >= TAIL_CUTOFF:
            with down:
                correction_low = gmpy2.mpfr(0)
            correction_high = tail_upper
            tail_magnitudes += 1
        else:
            # Both exp and log1p are monotone, so directed composition gives
            # a valid enclosure of this positive correction.
            with down:
                correction_low = gmpy2.log1p(gmpy2.exp(-a))
            with up:
                correction_high = gmpy2.log1p(gmpy2.exp(-a))
        with down:
            positive_low = a + correction_low
        with up:
            positive_high = a + correction_high
        bounds[bit] = (positive_low, positive_high)
        bounds[bit | 0x8000] = (correction_low, correction_high)

    def target(x):
        return max(x, mp.mpf(0)) + mp.log1p(mp.exp(-abs(x)))

    return core.build_reference_table(bounds, target, "mathematical_softplus", {
        "method": "For a=|x|, enclose c=log1p(exp(-a)) with directed exp and log1p. Softplus(a)=a+c and softplus(-a)=c; positive-side addition is rounded outward.",
        "tail_cutoff": TAIL_CUTOFF,
        "tail_bound_encodings": 2 * tail_magnitudes,
        "tail_correction_upper_bound": str(tail_upper),
        "tail_bound_argument": "For a>=32, 0<=log1p(exp(-a))<=exp(-a)<=exp(-32). The last value is computed with upward rounding. This avoids enormous integer-ratio allocations for extreme negative FP16 inputs.",
        "polynomial_independence": "The oracle uses mathematical exp and log1p, not the candidates' sqrt(2) range reduction, serialized constants, odd-power polynomial, or GPU arithmetic.",
    })


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all()
    return reference_table()[2][array.view(np.uint16)]


def plans():
    return core.plans(seed=136000)


def arrays(plan):
    return core.arrays(plan)


def cpu_audit(case_root):
    return core.cpu_audit(case_root, EXPECTED, reference_table(), ATOL, REFERENCE_BUDGET)


def validate(case_root):
    return core.validate(case_root, EXPECTED, reference_table(), "softplus", ATOL, REFERENCE_BUDGET, seed=136000)
