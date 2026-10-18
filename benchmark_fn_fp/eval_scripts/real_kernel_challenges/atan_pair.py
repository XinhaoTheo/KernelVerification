"""Independent directed-atan qualification for case_146 and case_147.

The oracle calls MPFR's mathematical atan with directed rounding. It does
not use the candidates' range reduction or rational coefficients.
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

EXPECTED = {"case_146": "trust", "case_147": "reject"}
ATOL = 1e-6
REFERENCE_BUDGET = 1e-10


@lru_cache(maxsize=1)
def reference_table():
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    bounds = {}
    for bit in core.finite_bit_patterns():
        value = float(np.array([bit], dtype=np.uint16).view(np.float16)[0])
        with down:
            low = gmpy2.atan(gmpy2.mpfr(value))
        with up:
            high = gmpy2.atan(gmpy2.mpfr(value))
        bounds[int(bit)] = (low, high)
    return core.build_reference_table(bounds, mp.atan, "mathematical_atan", {
        "method": "Direct MPFR atan at 320 bits, with RoundDown and RoundUp enclosing the exact mathematical value for each exact stored FP16 input.",
        "rational_independence": "Neither the Cephes coefficients nor the candidate range reduction, angle constants, polynomial evaluation or FP32 arithmetic is used by the independent oracle.",
    })


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all()
    return reference_table()[2][array.view(np.uint16)]


def plans():
    return core.plans(seed=146000)


def arrays(plan):
    return core.arrays(plan)


def cpu_audit(case_root):
    return core.cpu_audit(case_root, EXPECTED, reference_table(), ATOL, REFERENCE_BUDGET)


def validate(case_root):
    return core.validate(case_root, EXPECTED, reference_table(), "atan", ATOL, REFERENCE_BUDGET, seed=146000)
