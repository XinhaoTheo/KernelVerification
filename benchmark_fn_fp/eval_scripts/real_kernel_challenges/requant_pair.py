"""Independent integer requantization qualification for case_124/125.

The reference uses Python arbitrary-precision integers and the mathematical
two-round contract. It does not import or call candidate arithmetic helpers.
The finite checks below are distinguished from the algebraic correctness
argument: neither sampling nor a modeled identity proves GPU lowering correct.
Importing this module makes no network, model or GPU calls.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_124": "trust", "case_125": "reject"}
ATOL = 1e-6
MASK = (1 << 32) - 1
INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1


def scalar_reference(a, multiplier, shift):
    """Q31 high multiply rounds ties upward; POT divide ties away from zero."""
    product = int(a) * int(multiplier)
    high = (product + (1 << 30)) // (1 << 31)
    if int(shift) == 0:
        return high
    magnitude = (abs(high) + (1 << (int(shift) - 1))) // (1 << int(shift))
    return -magnitude if high < 0 else magnitude


def independent_reference(acc, multipliers, shifts, zero_points, scales):
    result = np.empty(acc.shape, dtype=np.float32)
    for channel in range(acc.shape[1]):
        multiplier, shift = int(multipliers[channel]), int(shifts[channel])
        zero, scale = int(zero_points[channel]), float(scales[channel])
        for row in range(acc.shape[0]):
            rounded = scalar_reference(acc[row, channel], multiplier, shift)
            # Big integers make the public add-then-clamp definition safe.
            quantized = min(127, max(-128, rounded + zero))
            result[row, channel] = (quantized - zero) * scale
    return result


def _modeled_limb(a, multiplier, shift, mutant=False):
    """Arithmetic model only; no candidate helper reuse or GPU execution."""
    product = int(a) * int(multiplier)
    low, high = product & MASK, product >> 32
    extra = 0 if shift == 0 else (1 << (30 + shift)) - ((1 << 31) if a < 0 else 0)
    low_extra, high_extra = extra & MASK, extra >> 32
    first = low + (1 << 30)
    second = (first & MASK) + low_extra
    carry = (first >> 32) + (((low + low_extra) >> 32) if mutant else (second >> 32))
    reconstructed = (high + high_extra + carry) * (1 << 32) + (second & MASK)
    return reconstructed >> (31 + shift)


def arithmetic_audit():
    """Record exact identities, plus independently reproducible finite checks."""
    rng = np.random.default_rng(124125)
    multipliers = [2**30 + x for x in (0, 1, 2, 127, 65535)]
    multipliers += [2**31 - x for x in (1, 2, 127, 65535)]
    multipliers += list(map(int, rng.integers(2**30, 2**31, size=4)))
    accumulators = list(range(-512, 513)) + [INT_MIN, INT_MIN + 1, INT_MAX - 1, INT_MAX]
    count, bad_visible = 0, 0
    for a in accumulators:
        for m in multipliers:
            for r in range(32):
                reference = scalar_reference(a, m, r)
                assert _modeled_limb(a, m, r) == reference, (a, m, r)
                wrong = _modeled_limb(a, m, r, mutant=True)
                bad_visible += min(127, max(-128, wrong)) != min(127, max(-128, reference))
                count += 1
    # Carries only change at a finite set of interval boundaries. Check both
    # ends of every interval for the two possible low-word extra biases.
    carry_count = 0
    for extra in (0, 1 << 31):
        boundaries = {0, 1 << 32}
        for bias in (1 << 30, extra, (1 << 30) + extra):
            if 0 < (1 << 32) - bias < (1 << 32):
                boundaries.add((1 << 32) - bias)
        points = sorted(boundaries)
        for left, right in zip(points[:-1], points[1:]):
            for low in {left, right - 1}:
                first = low + (1 << 30)
                second = (first & MASK) + extra
                carry = (first >> 32) + (second >> 32)
                assert carry * (1 << 32) + (second & MASK) == low + (1 << 30) + extra
                carry_count += 1
    witness = {"accumulator": 1, "multiplier": 1073741825, "shift": 1,
               "zero_point": 0, "scale": 0.0625, "reference_q": 1,
               "mutant_q": 0, "reference_output": 0.0625, "mutant_output": 0.0}
    assert scalar_reference(1, 1073741825, 1) == 1
    assert _modeled_limb(1, 1073741825, 1, True) == 0
    return {"passed": True, "modeled_integer_comparisons": count,
            "modeled_mutant_visible_mismatches": int(bad_visible),
            "carry_partition_endpoint_checks": carry_count, "witness": witness,
            "algebraic_argument": [
                "P=a*m lies in [-2^62+2^31, (2^31-1)^2], because a is int32 and 2^30<=m<2^31. The MIN*MIN saturation case is excluded by positive m.",
                "The public first rounding is H=floor((P+2^30)/2^31). For r>0 the second is floor((H+2^(r-1)-[H<0])/2^r). Nested floor division gives floor((P+2^30+2^(30+r)-[H<0]*2^31)/2^(31+r)).",
                "[H<0] can be replaced by [a<0]: their only disagreement is negative P with H=0, and both final quotients are zero for every r>=1. r=0 is handled separately.",
                "Writing P=hi*2^32+lo, each unsigned low-word addition has sum=(sum mod 2^32)+carry*2^32. Adding its carry to hi therefore reconstructs the exact biased integer. The second carry must depend on the first updated low word.",
                "The biased integer stays in [-2^62+3*2^30,3*2^61-3*2^30+1], so its high word lies in [-2^30,3*2^29-1] and each high addition fits int32. For r>=1 its arithmetic shift by 31+r equals the signed high word shifted by r-1; for r=0 combine high<<1 with low>>31, obtaining H in [-2^31+1,2^31-2].",
                "Clipping the intermediate Q to [-255,255] before adding z in [-128,127] preserves clip(Q+z,-128,127), including endpoints, and prevents int32 addition overflow.",
                "Final quantized-minus-zero lies in [-255,255]; multiplying it by the allowed power-of-two scales is exactly representable in FP32. No floating numerical tolerance is needed for the mathematical output."
            ],
            "scope": "Algebra covers the modeled scalar arithmetic over the complete public value domain. Endpoint/sampled checks cross-check that argument; they are not exhaustive enumeration of int32*Q31 inputs or a proof of compiler, tensor indexing, memory safety, or GPU behavior."}


def plans():
    records = [dict(rows=m, channels=n, style="random", seed=124000 + i)
               for i, (m, n) in enumerate(((1, 1), (2, 3), (7, 31), (31, 32),
                                          (33, 65), (257, 127), (1025, 128),
                                          (1, 257), (2, 128), (129, 257),
                                          (128, 129), (257, 1), (127, 127),
                                          (128, 128), (4096, 257)))]
    for i, style in enumerate(("small", "extremes", "thresholds", "carry_boundaries", "storage_offset")):
        records.append(dict(rows=1025, channels=128, style=style, seed=125000 + i))
    records.append(dict(rows=1, channels=1, style="witness", seed=125100))
    records.append(dict(rows=33, channels=65, style="input_alias", seed=125101))
    return records


def arrays(plan):
    rng = np.random.default_rng(plan["seed"])
    rows, channels = plan["rows"], plan["channels"]
    acc = rng.integers(INT_MIN, INT_MAX + 1, size=(rows, channels), dtype=np.int64).astype(np.int32)
    multiplier = rng.integers(2**30, 2**31, size=channels, dtype=np.int64).astype(np.int32)
    shift = (np.arange(channels) % 32).astype(np.int32)
    zero = np.resize(np.array([-128, 127, 0, -1, 1, -64, 64], dtype=np.int32), channels)
    scale = np.exp2(-(np.arange(channels) % 9)).astype(np.float32)
    style = plan["style"]
    if channels >= 128:
        # Every shift gets endpoint multipliers as well as arbitrary low bits.
        multiplier[:32] = 2**30
        multiplier[32:64] = 2**30 + 1
        multiplier[64:96] = 2**31 - 1
    if style in {"small", "storage_offset"}:
        acc[:] = np.arange(rows, dtype=np.int64)[:, None] % 1025 - 512
    elif style == "extremes":
        values = [INT_MIN, INT_MIN + 1, INT_MIN + 2, -2**30, -65536, -1, 0,
                  1, 65535, 2**30, INT_MAX - 2, INT_MAX - 1, INT_MAX]
        acc[:] = np.resize(np.array(values, dtype=np.int32), rows)[:, None]
    elif style == "thresholds":
        # Solve backwards near second-round transitions, then perturb legal a.
        for col in range(channels):
            m, r = int(multiplier[col]), int(shift[col])
            values = []
            for q in range(-130, 131):
                h = q * (1 << r) + ((1 << (r - 1)) if r else 0)
                center = ((h << 31) - (1 << 30)) // m
                values.extend(min(INT_MAX, max(INT_MIN, center + delta)) for delta in (-2, -1, 0, 1, 2))
            acc[:, col] = np.resize(np.array(values, dtype=np.int32), rows)
    elif style == "carry_boundaries":
        # Odd multiplier makes modular multiplication invertible, yielding
        # exact low-product values at all carry-change boundaries.
        multiplier |= 1
        lows = [((k << 30) + delta) & MASK for k in range(4) for delta in (-2, -1, 0, 1, 2)]
        for col in range(channels):
            inverse = pow(int(multiplier[col]), -1, 1 << 32)
            values = []
            for low in lows:
                value = (low * inverse) & MASK
                values.append(value if value < 2**31 else value - 2**32)
            acc[:, col] = np.resize(np.array(values, dtype=np.int32), rows)
    elif style == "witness":
        acc.fill(1); multiplier.fill(1073741825); shift.fill(1)
        zero.fill(0); scale.fill(0.0625)
    elif style == "input_alias":
        acc[0] = multiplier
        zero[:] = shift
    return acc, multiplier, shift, zero, scale


def _public_cpu(case_root, name):
    path = case_root / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"validate_inputs", "reference", "make_inputs"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def cpu_audit(case_root):
    arithmetic = arithmetic_audit()
    public_checks = []
    for name in EXPECTED:
        public = _public_cpu(Path(case_root), name)
        comparisons = 0
        for plan in plans():
            values = arrays(plan)
            inputs = tuple(torch.from_numpy(value.copy()) for value in values)
            public["validate_inputs"](*inputs, require_cuda=False)
            actual = public["reference"](*inputs).numpy()
            expected = independent_reference(*values)
            assert np.array_equal(actual, expected), (name, plan, float(np.max(np.abs(actual - expected))))
            comparisons += expected.size
        public_checks.append({"case": name, "plans": len(plans()), "elements": comparisons,
                              "public_reference_exact_match": True})
    return {"passed": True, "arithmetic": arithmetic, "public_references": public_checks}


def _load(case_root, name):
    modname = "requant_qualification_" + name
    spec = importlib.util.spec_from_file_location(modname, case_root / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root: Path):
    cpu = cpu_audit(case_root)
    cases = {}
    for name, expected_label in EXPECTED.items():
        module = _load(Path(case_root), name)
        records = []
        for plan in plans():
            values = arrays(plan)
            target = independent_reference(*values)
            inputs, backings = [], []
            for index, value in enumerate(values):
                tensor = torch.from_numpy(value.copy()).cuda()
                if plan["style"] == "storage_offset":
                    backing = torch.full((tensor.numel() + 29,), 37, device="cuda", dtype=tensor.dtype)
                    offset = 3 + index
                    view = backing[offset:offset + tensor.numel()].reshape(tensor.shape)
                    view.copy_(tensor)
                    tensor = view
                    assert tensor.storage_offset() == offset
                else:
                    backing = tensor
                inputs.append(tensor); backings.append(backing)
            if plan["style"] == "input_alias":
                inputs[1] = inputs[0][0]
                inputs[3] = inputs[2]
                backings = inputs.copy()
                assert inputs[0].untyped_storage().data_ptr() == inputs[1].untyped_storage().data_ptr()
                assert inputs[2].untyped_storage().data_ptr() == inputs[3].untyped_storage().data_ptr()
            module.validate_inputs(*inputs)
            before = [tensor.clone() for tensor in backings]
            actual = module.run(*inputs)
            torch.cuda.synchronize()
            valid_type = isinstance(actual, torch.Tensor) and actual.dtype == torch.float32 and actual.device == inputs[0].device and tuple(actual.shape) == target.shape
            assert valid_type, (name, plan, "output shape/dtype/device")
            observed = actual.cpu().numpy()
            public = module.reference(*inputs).cpu().numpy()
            assert np.array_equal(public, target), (name, plan, "public reference drift")
            preserved = all(torch.equal(old.view(torch.uint8), new.view(torch.uint8)) for old, new in zip(before, backings))
            output_storage = actual.untyped_storage().data_ptr()
            fresh = all(output_storage != tensor.untyped_storage().data_ptr() for tensor in inputs)
            finite = bool(np.isfinite(observed).all())
            delta = np.abs(observed.astype(np.float64) - target.astype(np.float64))
            error = float(np.max(delta))
            witness = None
            if not finite or error > ATOL:
                location = np.unravel_index(np.argmax(np.where(np.isfinite(delta), delta, np.inf)), delta.shape)
                row, col = map(int, location)
                witness = {"index": [row, col], "accumulator": int(values[0][row, col]),
                           "multiplier": int(values[1][col]), "shift": int(values[2][col]),
                           "zero_point": int(values[3][col]), "scale": float(values[4][col]),
                           "actual": float(observed[row, col]), "reference": float(target[row, col]),
                           "absolute_error": float(delta[row, col]), "allowed_error": ATOL}
            records.append(dict(plan, legal=True, passed=bool(valid_type and finite and preserved and fresh and error <= ATOL),
                                exact_match=bool(np.array_equal(observed, target)), max_abs_error=error,
                                max_error_ratio=error / ATOL, finite=finite, inputs_preserved=preserved,
                                output_fresh=fresh, shape_dtype_valid=valid_type,
                                storage_offsets=[int(t.storage_offset()) for t in inputs], witness=witness))
        failures = [record for record in records if not record["passed"]]
        cases[name] = {"ground_truth": expected_label, "tests": records, "tests_total": len(records),
                       "tests_passed": len(records) - len(failures), "counterexamples": failures,
                       "qualification_passed": bool(failures) if expected_label == "reject" else not failures and cpu["passed"],
                       "label_scope": "Fixed implementation on the complete public integer requantization/dequantization contract.",
                       "analysis": "Independent Python-bigint oracle, separately stated full-domain scalar arithmetic argument, finite carry-partition cross-check, and GPU arithmetic/interface checks. GPU tests alone do not establish universal program correctness."}
    return {"cases": cases, "cpu_audit": cpu, "qualification_passed": all(case["qualification_passed"] for case in cases.values())}
