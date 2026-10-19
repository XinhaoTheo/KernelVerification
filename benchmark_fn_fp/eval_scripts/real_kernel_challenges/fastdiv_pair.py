"""Exact integer qualification for uint32 divmod-by-seven cases 140/141.

Python's arbitrary-precision divmod is the independent oracle. A complete
quotient/remainder partition proves the correct scalar formula and identifies
the other formula's entire failing set. GPU checks separately exercise actual
Triton lowering and tensor properties; they are not a 2**32-value enumeration.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_67": "trust", "case_68": "reject"}
DIVISOR = 7
SHIFT = 34
BASE = 1 << SHIFT
UINT_MAX = (1 << 32) - 1
DOWN_MULTIPLIER = BASE // DIVISOR
UP_MULTIPLIER = (BASE + DIVISOR - 1) // DIVISOR
ATOL = 0
RTOL = 0


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def independent_reference(values):
    values = np.asarray(values)
    assert values.dtype == np.uint32 and values.ndim == 1
    return np.array([divmod(int(value), DIVISOR) for value in values], dtype=np.uint32)


def modeled_candidate(value, name):
    """The audited scalar formulas with explicit uint32 remainder conversion."""
    n = int(value)
    assert 0 <= n <= UINT_MAX and name in EXPECTED
    if name == "case_67":
        product = (n + 1) * DOWN_MULTIPLIER
    else:
        product = n * UP_MULTIPLIER
    assert 0 <= product < (1 << 64)
    q = product >> SHIFT
    return q, (n - q * DIVISOR) & UINT_MAX


def arithmetic_audit():
    """Seven linear remainder classes cover every possible uint32 input.

    Write n=7*q+r, r in [0,6]. For each fixed r, the residual numerator is
    linear in q, so exact endpoint inequalities prove its whole interval.
    This is an integer argument, not a sampled confidence statement.
    """
    assert DIVISOR * DOWN_MULTIPLIER + 2 == BASE
    assert DIVISOR * UP_MULTIPLIER - 5 == BASE
    assert DOWN_MULTIPLIER * (UINT_MAX + 1) < (1 << 64)
    assert UP_MULTIPLIER * UINT_MAX < (1 << 64)
    rows, full_size, failing_count = [], 0, 0
    failures = []
    for remainder in range(DIVISOR):
        q_max = (UINT_MAX - remainder) // DIVISOR
        full_size += q_max + 1
        # m_down*(n+1) = BASE*q + m_down*(r+1) - 2*q.
        good_low = DOWN_MULTIPLIER * (remainder + 1) - 2 * q_max
        good_high = DOWN_MULTIPLIER * (remainder + 1)
        assert 0 <= good_low <= good_high < BASE
        # m_up*n = BASE*q + 5*q + m_up*r.
        up_low = UP_MULTIPLIER * remainder
        up_high = 5 * q_max + up_low
        assert 0 <= up_low <= up_high < 2 * BASE
        first_bad_q = max(0, (BASE - up_low + 4) // 5)
        count = max(0, q_max - first_bad_q + 1)
        failing_count += count
        failing_range = None
        if count:
            first = DIVISOR * first_bad_q + remainder
            last = DIVISOR * q_max + remainder
            failing_range = {"first_quotient": first_bad_q, "last_quotient": q_max,
                             "first_input": first, "last_input": last,
                             "input_stride": DIVISOR, "count": count}
            failures.append(failing_range)
            assert modeled_candidate(first, "case_68") != divmod(first, DIVISOR)
            assert modeled_candidate(last, "case_68") != divmod(last, DIVISOR)
            if first_bad_q > 0:
                before = DIVISOR * (first_bad_q - 1) + remainder
                assert modeled_candidate(before, "case_68") == divmod(before, DIVISOR)
        rows.append({"remainder": remainder, "q_min": 0, "q_max": q_max,
                     "number_of_inputs": q_max + 1,
                     "round_down_residual_min": good_low,
                     "round_down_residual_max": good_high,
                     "round_up_residual_min": up_low,
                     "round_up_residual_max": up_high,
                     "round_up_failing_range": failing_range})
    assert full_size == (1 << 32) and failing_count > 0
    signed_domain_qmax = ((1 << 31) - 1) // DIVISOR
    signed_domain_residual_upper = 5 * signed_domain_qmax + 6 * UP_MULTIPLIER
    assert signed_domain_residual_upper < BASE
    first = min(row["first_input"] for row in failures)
    q_actual, r_actual = modeled_candidate(first, "case_68")
    q_expected, r_expected = divmod(first, DIVISOR)
    return {"passed": True, "covered_scalar_inputs": full_size,
            "proof_method": "Exact quotient/remainder partition with monotone affine residuals; seven endpoint-checked intervals cover all uint32 values.",
            "divisor": DIVISOR, "shift": SHIFT, "power_of_two": BASE,
            "round_down_multiplier": DOWN_MULTIPLIER,
            "round_up_multiplier": UP_MULTIPLIER,
            "round_down_max_product": DOWN_MULTIPLIER * (UINT_MAX + 1),
            "round_up_max_product": UP_MULTIPLIER * UINT_MAX,
            "uint64_max": (1 << 64) - 1,
            "partitions": rows, "round_up_failure_count_full_domain": failing_count,
            "round_up_valid_nonnegative_int32": True,
            "signed_domain_residual_upper_bound": signed_domain_residual_upper,
            "round_up_first_failure": {"input": first, "actual": [q_actual, r_actual],
                                       "reference": [q_expected, r_expected]},
            "algebraic_argument": [
                "Every uint32 n has one representation n=7*q+r, 0<=r<=6, 0<=q<=floor((UINT_MAX-r)/7). The seven classes are disjoint and their sizes sum to 2^32.",
                "For m=floor(2^34/7), 7*m=2^34-2. Hence m*(n+1)=2^34*q+[m*(r+1)-2*q]. In each remainder class the bracket decreases with q, and its exact endpoints satisfy 0<=bracket<2^34. Shifting by34 therefore returns exactly q for every uint32 n.",
                "Widening n to uint64 before n+1 preserves the UINT_MAX endpoint. The largest product m*(UINT_MAX+1) is below2^64. Correct q*7<=n, so subtraction returns the exact remainder in0..6 and output uint32 casts are exact.",
                "For M=ceil(2^34/7), 7*M=2^34+5. Thus M*n=2^34*q+[5*q+M*r]. In each remainder class the bracket increases with q and stays below2*2^34. The approximate quotient is q+1 precisely when bracket>=2^34; thresholds recorded for every r give the complete failing set, not a sampled estimate.",
                "For all nonnegative int32 inputs, 5*floor((2^31-1)/7)+6*M<2^34, so the round-up formula is exact on that narrower domain. Its synthetic uint32 adaptation extends beyond that guarantee.",
                "If the round-up quotient is q+1, subtracting7*(q+1) from n yields r-7. Conversion to uint32 gives its modulo2^32 representation, which violates the required exact remainder0..6."
            ],
            "scope": "Complete scalar integer proof for the audited expressions and widths; does not by itself prove compiler lowering, scheduling or arbitrary tensor memory behavior."}


def _boundary_values():
    proof = arithmetic_audit()
    numbers = {0, 1, 6, 7, 8, UINT_MAX - 1, UINT_MAX, (1 << 31) - 1, 1 << 31}
    for bit in range(33):
        center = 1 << bit
        for delta in range(-8, 9):
            if 0 <= center + delta <= UINT_MAX:
                numbers.add(center + delta)
    for row in proof["partitions"]:
        quotients = {0, 1, row["q_max"], max(0, row["q_max"] - 1)}
        bad = row["round_up_failing_range"]
        if bad:
            quotients.update(bad["first_quotient"] + delta for delta in range(-3, 4))
        for q in quotients:
            for r in range(DIVISOR):
                value = DIVISOR * q + r
                if 0 <= value <= UINT_MAX:
                    numbers.add(value)
    return np.array(sorted(numbers), dtype=np.uint32)


def plans():
    lengths = [1, 2, 3, 31, 32, 33, 127, 128, 129, 255, 256, 257,
               511, 512, 513, 1023, 1024, 1025, 4093, 4096, 4097,
               65535, 65536, 65537, 131071, 131072]
    rows = [dict(n=n, style="random", seed=140000 + i, offset=0)
            for i, n in enumerate(lengths)]
    rows += [dict(n=65536, style="low_block", seed=140100, offset=0),
             dict(n=65536, style="high_block", seed=140101, offset=0),
             dict(n=65537, style="boundary_values", seed=140102, offset=7),
             dict(n=1, style="first_failure", seed=140103, offset=13),
             dict(n=257, style="threshold_window", seed=140104, offset=3),
             dict(n=131072, style="last_remainder", seed=140105, offset=17),
             dict(n=17, style="boundary_values", seed=140106, offset=1)]
    return rows


def arrays(plan):
    n, style = int(plan["n"]), plan["style"]
    rng = np.random.default_rng(plan["seed"])
    first = arithmetic_audit()["round_up_first_failure"]["input"]
    if style == "low_block":
        values = np.arange(n, dtype=np.uint64)
    elif style == "high_block":
        values = np.arange(UINT_MAX + 1 - n, UINT_MAX + 1, dtype=np.uint64)
    elif style == "boundary_values":
        values = np.resize(_boundary_values(), n)
    elif style == "first_failure":
        values = np.full(n, first, dtype=np.uint64)
    elif style == "threshold_window":
        values = np.arange(first - n // 2, first - n // 2 + n, dtype=np.uint64)
    elif style == "last_remainder":
        q = rng.integers(0, (UINT_MAX - 6) // DIVISOR + 1, size=n, dtype=np.uint64)
        values = q * DIVISOR + 6
    else:
        values = rng.integers(0, UINT_MAX + 1, size=n, dtype=np.uint64)
    values = values.astype(np.uint32)
    assert values.shape == (n,)
    return values


def _public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"validate_inputs", "reference", "make_inputs"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _source_binding(case_root):
    records = {}
    formulas = {"case_67": "(value + 1) * MULTIPLIER >> SHIFT",
                "case_68": "value * MULTIPLIER >> SHIFT"}
    for name in EXPECTED:
        path = Path(case_root) / name / "kernel.py"
        tree = ast.parse(path.read_text())
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
        namespace = {}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
        expected_magic = DOWN_MULTIPLIER if name == "case_67" else UP_MULTIPLIER
        assert namespace["_DIVISOR"] == DIVISOR and namespace["_SHIFT"] == SHIFT
        assert namespace["_MULTIPLIER"] == expected_magic
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_unflatten")
        assign = {node.targets[0].id: node.value for node in kernel.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        expected = ast.parse(formulas[name], mode="eval").body
        assert ast.dump(assign["quotient"]) == ast.dump(expected)
        assert ast.unparse(assign["value"]).endswith(".to(tl.uint64)")
        assert ast.unparse(assign["remainder"]) == "value - quotient * 7"
        records[name] = {"kernel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "divisor": DIVISOR, "shift": SHIFT, "multiplier": expected_magic,
                         "quotient_expression": ast.unparse(assign["quotient"]),
                         "widen_before_arithmetic": "uint64",
                         "scope": "AST checks bind the scalar proof's constants/formulas/width to this source; actual GPU lowering remains separately tested."}
    return records


def cpu_audit(case_root):
    proof = arithmetic_audit()
    reference_checks, candidates = [], []
    for name in EXPECTED:
        public = _public_cpu(case_root, name)
        samples, mismatches = 0, 0
        for plan in plans():
            values = arrays(plan)
            x = torch.from_numpy(values.copy())
            public["validate_inputs"](x, require_cuda=False)
            reference = independent_reference(values)
            assert np.array_equal(public["reference"](x).numpy(), reference), (name, plan)
            candidate = np.array([modeled_candidate(value, name) for value in values], dtype=np.uint32)
            bad = np.any(candidate != reference, axis=1)
            samples += values.size
            mismatches += int(bad.sum())
        reference_checks.append({"case": name, "plans": len(plans()),
                                 "scalar_values": samples, "public_reference_exact_match": True})
        candidates.append({"case": name, "cpu_sampled_scalar_values": samples,
                           "cpu_mismatching_values": mismatches,
                           "scope": "CPU samples cross-check the complete partition proof; they do not constitute a full2^32 enumeration or GPU validation."})
    assert candidates[0]["cpu_mismatching_values"] == 0
    assert candidates[1]["cpu_mismatching_values"] > 0
    return {"passed": True, "arithmetic": proof, "public_references": reference_checks,
            "candidates": candidates, "source_binding": _source_binding(case_root),
            "indexing": {"block_size": 256, "max_n": 131072,
                         "argument": "Each program covers an adjacent disjoint256-index interval, masked by index<N. Active i writes exactly output[2*i] and[2*i+1]. Thus all2*N output coordinates are covered once, and maximum offset262143 is int32-safe. Contiguous input offsets do not alter pointer-relative indices.",
                         "scope": "Source arithmetic argument; GPU shape, tail, storage-offset and backing-guard checks are separate."},
            "oracle_dependency_hashes": dependency_hashes()}


def _load(case_root, name):
    spec = importlib.util.spec_from_file_location("fastdiv_qualification_" + name, Path(case_root) / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu = cpu_audit(case_root)
    cases = {}
    for name, label in EXPECTED.items():
        module = _load(case_root, name)
        records = []
        for plan in plans():
            values, offset = arrays(plan), int(plan["offset"])
            backing = torch.full((offset + values.size + 31,), 0x13579BDF, dtype=torch.uint32, device="cuda")
            x = backing[offset:offset + values.size]
            x.copy_(torch.from_numpy(values).cuda())
            before = backing.clone()
            assert x.is_contiguous() and x.storage_offset() == offset
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            target = independent_reference(values)
            valid = type(actual) is torch.Tensor and actual.dtype == torch.uint32 and actual.device == x.device and tuple(actual.shape) == target.shape
            assert valid, (name, plan, "output type/dtype/device/shape")
            observed = actual.cpu().numpy()
            public = module.reference(x).cpu().numpy()
            assert np.array_equal(public, target), (name, plan, "public reference")
            bad = np.any(observed != target, axis=1)
            numeric = not bad.any()
            preserved = bool(torch.equal(before.view(torch.uint8), backing.view(torch.uint8)))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            witness = None
            if bad.any():
                index = int(np.flatnonzero(bad)[0])
                witness = {"index": index, "input": int(values[index]),
                           "actual": list(map(int, observed[index])),
                           "reference": list(map(int, target[index])),
                           "comparison": "exact integer equality", "allowed_error": 0}
            delta = np.abs(observed.astype(np.int64) - target.astype(np.int64))
            records.append(dict(plan, legal=True, passed=bool(numeric and preserved and fresh),
                                numeric_passed=bool(numeric), exact_match=bool(numeric),
                                definite_failures=int(bad.sum()), elements=int(values.size),
                                max_abs_error=int(delta.max()), comparison="exact integer equality",
                                inputs_preserved=preserved, inputs_and_backing_guards_preserved=preserved,
                                output_fresh=bool(fresh), shape_dtype_valid=True,
                                storage_offset=int(x.storage_offset()), witness=witness))
        failures = [row for row in records if not row["passed"]]
        numeric_reject = any(row["definite_failures"] > 0 for row in records)
        cases[name] = {"ground_truth": label, "tests": records, "tests_total": len(records),
                       "tests_passed": len(records) - len(failures), "counterexamples": failures,
                       "qualification_passed": numeric_reject if label == "reject" else not failures and cpu["passed"],
                       "gpu_elements_compared": sum(row["elements"] for row in records),
                       "scalar_domain_inputs_proved": 1 << 32,
                       "comparison": "exact integer equality", "atol": 0, "rtol": 0,
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One fixed implementation on every uint32 input value; test inputs only supply witnesses.",
                       "analysis": "Independent arbitrary-precision integer divmod reference and complete quotient/remainder-partition proof, plus actual GPU tests of arithmetic, lengths/tails, offsets, input/backing preservation and fresh output. No claim of exhaustively executing every tensor on GPU."}
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": all(row["qualification_passed"] for row in cases.values())}
