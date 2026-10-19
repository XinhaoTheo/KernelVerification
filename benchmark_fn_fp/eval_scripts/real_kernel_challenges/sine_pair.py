"""Independent directed sine oracle for the fixed case_59/133 pair."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import re
import sys

import gmpy2
import mpmath as mp
import numpy as np
import torch

try:
    from . import adaptive_unary_oracle as core
except ImportError:
    import adaptive_unary_oracle as core

EXPECTED = {"case_59": "trust", "case_60": "reject"}
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
            lower = gmpy2.sin(gmpy2.mpfr(value))
        with up:
            upper = gmpy2.sin(gmpy2.mpfr(value))
        bounds[int(bit)] = (lower, upper)
    return core.build_reference_table(bounds, mp.sin, "mathematical_sine", {
        "method": "MPFR sin at 320-bit precision evaluated directly with downward/upward rounding on each exact stored FP16 input, including both signs, zeros, subnormals and extrema. The mpmath 110-digit crosscheck independently evaluates sin on all 63,488 encodings.",
        "argument_reduction_independence": "Neither candidate quadrant reduction, pi/2 split, polynomial coefficients, FMA path nor arithmetic helpers participate in the oracle.",
        "directed_mpfr_function_evaluations": 2 * len(bounds),
    })


def independent_reference(values):
    array = np.asarray(values)
    assert array.dtype == np.float16 and np.isfinite(array).all()
    return reference_table()[2][array.view(np.uint16)]


def plans():
    return core.plans(seed=132000)


def arrays(plan):
    return core.arrays(plan)


def cpu_audit(case_root):
    return core.cpu_audit(case_root, EXPECTED, reference_table(), ATOL, REFERENCE_BUDGET)


def _inspect_ptx():
    """Capture actual compiled arithmetic without exposing it to model arms.

    Called only inside the parent's GPU qualification, after core.validate has
    imported and exercised both public modules. This adds one three-element
    launch per case; it does not alter public files or the shared frozen helper.
    """
    records = {}
    for name in EXPECTED:
        module = sys.modules["sine_qualification_" + name]
        x = torch.tensor([63168.0, -63168.0, 0.25], dtype=torch.float16, device="cuda")
        before = x.clone()
        output = torch.empty_like(x, dtype=torch.float32)
        compiled = module._sine_kernel[(1,)](
            x, output, N=3, BLOCK=256, num_warps=4, enable_fp_fusion=False)
        torch.cuda.synchronize()
        ptx = compiled.asm["ptx"]
        assert isinstance(ptx, str) and ptx
        # All arithmetic apart from the deliberately explicit reduction FMA
        # uses ordinary operators with automatic contraction disabled.
        fma_count = len(re.findall(r"\bfma(?:\.[a-z0-9]+)*\.f32\b", ptx))
        multiply_count = len(re.findall(r"\bmul(?:\.[a-z0-9]+)*\.f32\b", ptx))
        add_sub_count = len(re.findall(r"\b(?:add|sub)(?:\.[a-z0-9]+)*\.f32\b", ptx))
        expected_fma = name == "case_59"
        lowering_passed = fma_count > 0 if expected_fma else fma_count == 0
        preserved = bool(torch.equal(x.view(torch.uint8), before.view(torch.uint8)))
        bits = x.cpu().numpy().view(np.uint16)
        numeric = core.compare(output.cpu().numpy(), bits, reference_table(), ATOL)
        # Every extra legal execution is also evidence about a trust label.
        # A reject case already has its required counterexample in the main
        # qualification; these additional values need not all violate.
        numeric_label_ok = numeric["numeric_passed"] if EXPECTED[name] == "trust" else True
        records[name] = {
            "passed": bool(lowering_passed and preserved and numeric_label_ok),
            "expected_explicit_fma": expected_fma,
            "fma_f32_instruction_count": fma_count,
            "mul_f32_instruction_count": multiply_count,
            "add_sub_f32_instruction_count": add_sub_count,
            "enable_fp_fusion": False,
            "n": 3, "block": 256, "num_warps": 4,
            "candidate_gpu_launches": 1, "candidate_elements": 3,
            "input_preserved": preserved,
            "numeric_result": numeric,
            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
            "ptx": ptx,
            "scope": "Actual N=3 compiled specialization. N is constexpr, so this is not a claim to inspect every N specialization. The source scalar arithmetic is independent of N, which only controls the active-lane mask.",
        }
    assert all(row["passed"] for row in records.values()), {
        name: {k: v for k, v in row.items() if k != "ptx"}
        for name, row in records.items()}
    return records


def validate(case_root):
    result = core.validate(case_root, EXPECTED, reference_table(), "sine", ATOL, REFERENCE_BUDGET, seed=132000)
    result["ptx_inspection"] = _inspect_ptx()
    result["qualification_passed"] = result["qualification_passed"] and all(
        row["passed"] for row in result["ptx_inspection"].values())
    return result
