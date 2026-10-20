"""Independent real cube-root qualification for case_101 and case_102."""
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

EXPECTED = {"case_101": "trust", "case_102": "reject"}
RTOL = 1e-6
REFERENCE_RELATIVE_BUDGET = 1e-12


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest(),
            **core.dependency_hashes()}


def legal_bits():
    return core.finite_bit_patterns()


def tensor_from_bits(bits):
    return torch.from_numpy(np.asarray(bits, dtype=np.uint16).view(np.float16).copy())


@lru_cache(maxsize=1)
def reference_table():
    bounds = {}
    down = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundDown)
    up = gmpy2.context(precision=core.ORACLE_BITS, round=gmpy2.RoundUp)
    for bit in legal_bits():
        value = float(np.array([bit], dtype=np.uint16).view(np.float16)[0])
        with down:
            lower = gmpy2.cbrt(gmpy2.mpfr(value))
        with up:
            upper = gmpy2.cbrt(gmpy2.mpfr(value))
        bounds[int(bit)] = (lower, upper)
    return core.build_reference_table(
        bounds, lambda x: mp.sign(x) * mp.root(abs(x), 3), "mathematical_real_cbrt",
        {"method": "Direct MPFR real cbrt with RoundDown and RoundUp; outward binary64 endpoints. Independent mpmath real cube root crosschecks every finite FP16 encoding.",
         "independence": "No candidate exponent splitting, polynomial coefficients, Newton iteration, remainder constants or FP32 reconstruction is used by the oracle.",
         "oracle_dependency_hashes": dependency_hashes()})


def compare(observed, bits, relative_tolerance=RTOL):
    lower, upper, center, _ = reference_table()
    y = np.asarray(observed, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint16)
    assert y.shape == bits.shape and y.ndim == 1 and y.size > 0
    lo, hi, ref = lower[bits], upper[bits], center[bits]
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    zero = (bits & 0x7FFF) == 0
    minimum_magnitude = np.where(zero, 1.0, np.minimum(np.abs(lo), np.abs(hi)))
    maximum_magnitude = np.where(zero, 1.0, np.maximum(np.abs(lo), np.abs(hi)))
    finite = np.isfinite(y)
    far = np.nextafter(np.maximum(np.abs(y - lo), np.abs(y - hi)), np.inf)
    near = np.maximum(np.nextafter(np.maximum(np.maximum(lo - y, y - hi), 0.0), -np.inf), 0.0)
    worst = np.nextafter(far / minimum_magnitude, np.inf)
    best = np.maximum(np.nextafter(near / maximum_magnitude, -np.inf), 0.0)
    worst = np.where(zero, np.where(y == 0.0, 0.0, np.inf), worst)
    best = np.where(zero, np.where(y == 0.0, 0.0, np.inf), best)
    failing = (~finite) | (best > relative_tolerance)
    ambiguous = finite & (best <= relative_tolerance) & (worst > relative_tolerance)
    index = int(np.argmax(np.where(finite, worst, np.inf)))
    witness = None
    if failing.any() or ambiguous.any():
        witness = {
            "index": index, "input_bits_hex": f"0x{int(bits[index]):04x}",
            "input": float(bits[index:index + 1].view(np.float16)[0]), "actual": float(y[index]),
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
        "max_relative_error": float(np.max(np.abs(y - ref) / np.where(zero, 1.0, np.abs(ref)))),
        "max_relative_error_upper_bound": maximum,
        "max_error_ratio": maximum / relative_tolerance,
        "minimum_relative_tolerance_margin": relative_tolerance - maximum,
        "worst_input_bits_hex": f"0x{int(bits[index]):04x}",
        "worst_input": float(bits[index:index + 1].view(np.float16)[0]), "witness": witness,
    }



def plans(seed=101000):
    lengths = (1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257,
               511, 512, 513, 1023, 1024, 1025, 4095, 4096, 4097,
               63487, 63489, 65535, 65536, 65537, 131071, 131072)
    rows = [dict(n=n, style="random_bits", seed=seed + i, offset=0)
            for i, n in enumerate(lengths)]
    for i, style in enumerate(("all_legal", "all_legal_reverse", "all_legal_shuffle")):
        rows.append(dict(n=63488, style=style, seed=seed + 100 + i, offset=0))
    rows.extend([
        dict(n=63488, style="all_legal_shuffle", seed=seed + 110, offset=7),
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
                    0x4000, 0x4001, 0x43FF, 0x4400, 0x4401, 0x7BFE, 0x7BFF]
        boundary = np.array(positive + [b | 0x8000 for b in positive ],
                            dtype=np.uint16)
        bits = np.resize(boundary, plan["n"])
    elif style == "subnormals":
        bits = bits[((bits & 0x7C00) == 0) & ((bits & 0x3FF) != 0)]
    elif style == "signed_zeros":
        bits = np.resize(np.array([0, 0x8000], dtype=np.uint16), plan["n"])
    else:
        bits = rng.choice(bits, size=plan["n"], replace=True)
    assert bits.size == plan["n"] and bits.dtype == np.uint16
    assert np.isfinite(bits.view(np.float16)).all()
    return bits.view(np.float16).copy()


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
        invalid_inputs = (
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
        records.append({"case": name, "public_reference_dtype": str(observed.dtype),
                        "invalid_input_checks": len(invalid_inputs), **result})
    return {"passed": True, "oracle": reference_table()[3],
            "indexing": core.indexing_audit(), "public_references": records,
            "oracle_dependency_hashes": dependency_hashes(),
            "scope": "Exhaustive FP16 domain reference audit, separate from GPU arithmetic qualification."}


def validate(case_root):
    cpu = cpu_audit(case_root)
    cases = {}
    for name, label in EXPECTED.items():
        module = core._load(case_root, name, "cbrt")
        records = []
        for plan in plans():
            bits, offset = arrays(plan).view(np.uint16), int(plan["offset"])
            backing = torch.full((offset + bits.size + 23,), 0.3125, dtype=torch.float16, device="cuda")
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
            "scalar_domain_encodings": 63488,
            "exhaustive_scalar_passes": sum(row["exhaustive_scalar_domain"] and row["passed"] for row in records),
            "gpu_elements_compared": sum(row["elements"] for row in records),
            "oracle_dependency_hashes": dependency_hashes(),
            "label_scope": "One fixed implementation over the full public domain; inputs only supply witnesses.",
            "analysis": "All 63,488 finite FP16 encodings compared with directed mathematical intervals in multiple orders and offsets. Other N/tails, input guards and fresh output are checked separately. Scalar coverage is not exhaustive GPU execution coverage.",
        }
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": all(row["qualification_passed"] for row in cases.values())}
