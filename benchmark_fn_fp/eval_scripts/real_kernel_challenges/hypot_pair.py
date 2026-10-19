"""Directed mathematical qualification for bounded normalization, cases134/135.

The oracle evaluates x/sqrt(1+x*x) directly with directed MPFR arithmetic;
it shares neither the inverse-square-root seed nor Newton arithmetic with
the candidates. Candidate truth is fixed over the entire public domain.
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

EXPECTED = {"case_61": "trust", "case_62": "reject"}
ATOL = 1e-6
REFERENCE_BUDGET = 1e-10


@lru_cache(maxsize=1)
def reference_table():
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    bounds = {}
    for bit in range(0x7C00):
        value = float(np.array([bit], dtype=np.uint16).view(np.float16)[0])
        with down:
            a = gmpy2.mpfr(value)
            squared_low = 1 + a * a
            denominator_low = gmpy2.sqrt(squared_low)
        with up:
            squared_high = 1 + a * a
            denominator_high = gmpy2.sqrt(squared_high)
        with down:
            positive_low = a / denominator_high
        with up:
            positive_high = a / denominator_low
        bounds[bit] = (positive_low, positive_high)
        with down:
            negative_low = -positive_high
        with up:
            negative_high = -positive_low
        bounds[bit | 0x8000] = (negative_low, negative_high)

    return core.build_reference_table(
        bounds, lambda x: x / mp.sqrt(1 + x * x),
        "mathematical_bounded_normalization", {
            "method": "For a=|x|, enclose 1+a*a, its square root, and a/sqrt(1+a*a) with directed MPFR rounding. Division pairs the lower numerator with the upper denominator and conversely. Negating the interval handles negative inputs.",
            "candidate_independence": "The oracle uses mathematical square root and division, not the candidates' bit seed, Newton steps, constants, or arithmetic helpers.",
        })


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all()
    return reference_table()[2][array.view(np.uint16)]


def plans():
    return core.plans(seed=134000)


def arrays(plan):
    return core.arrays(plan)


def cpu_audit(case_root):
    return core.cpu_audit(case_root, EXPECTED, reference_table(), ATOL, REFERENCE_BUDGET)


def validate(case_root):
    return core.validate(case_root, EXPECTED, reference_table(), "hypot", ATOL, REFERENCE_BUDGET, seed=134000)
