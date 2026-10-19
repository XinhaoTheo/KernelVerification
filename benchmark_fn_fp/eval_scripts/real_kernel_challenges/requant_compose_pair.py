"""Exact independent qualification for single-rounding Q31 requantization.

The oracle rounds Python-bigint rational quotients by comparing remainders.
It does not reuse the candidate or public-reference bias/shift expression.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_83": "trust", "case_84": "reject"}
ATOL = RTOL = 0
INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def scalar_reference(a, m, r, z):
    denominator = 2 ** (31 + int(r))
    quotient, remainder = divmod(int(a) * int(m), denominator)
    if 2 * remainder >= denominator:
        quotient += 1
    return min(127, max(-128, quotient + int(z)))


def independent_reference(a, m, r, z):
    return np.fromiter((scalar_reference(a[row, col], m[col], r[col], z[col])
                        for row in range(a.shape[0]) for col in range(a.shape[1])),
                       dtype=np.int8, count=a.size).reshape(a.shape)


def plans():
    result = [dict(rows=rows, channels=channels, style="random", offset=i % 7,
                   seed=156000 + i)
              for i, (rows, channels) in enumerate(((1, 1), (2, 3), (7, 31),
                  (31, 32), (33, 65), (257, 127), (1, 257), (129, 257),
                  (128, 129), (257, 1), (127, 127), (128, 128), (4096, 257)))]
    for i, style in enumerate(("small", "extremes", "thresholds", "factor_boundaries")):
        result.append(dict(rows=1025, channels=128, style=style,
                           offset=3 + i, seed=157000 + i))
    result.append(dict(rows=1, channels=1, style="witness", offset=7, seed=157100))
    result.append(dict(rows=33, channels=65, style="input_alias", offset=0, seed=157101))
    return result


def arrays(plan):
    rng = np.random.default_rng(plan["seed"])
    rows, channels = plan["rows"], plan["channels"]
    a = rng.integers(INT_MIN, INT_MAX + 1, (rows, channels), dtype=np.int64).astype(np.int32)
    m = rng.integers(1 << 30, 1 << 31, channels, dtype=np.int64).astype(np.int32)
    r = (np.arange(channels) % 32).astype(np.int32)
    z = np.resize(np.array([-128, 127, 0, -1, 1, -64, 64], dtype=np.int32), channels)
    style = plan["style"]
    if style == "small":
        a[:] = np.arange(rows, dtype=np.int32)[:, None] - 512
    elif style == "extremes":
        values = [INT_MIN, INT_MIN + 1, INT_MAX, INT_MAX - 1, -1, 0, 1,
                  -(1 << 30), 1 << 30, (1 << 30) + 1]
        a[:] = np.resize(np.array(values, dtype=np.int32), rows)[:, None]
        m[:] = np.resize(np.array([1 << 30, (1 << 30) + 1,
                                  INT_MAX, INT_MAX - 1], dtype=np.int32), channels)
    elif style == "thresholds":
        for col in range(channels):
            values = []
            denominator = 1 << (31 + int(r[col]))
            for output in range(-256, 257):
                center = ((2 * output + 1) * denominator) // (2 * int(m[col]))
                values.extend(min(INT_MAX, max(INT_MIN, center + delta)) for delta in (-1, 0, 1))
            a[:, col] = np.resize(np.asarray(values, dtype=np.int32), rows)
    elif style == "factor_boundaries":
        for col in range(channels):
            # (2^k+d)(2^k-d) is near an exact power of two. These are legal
            # integer products, with no reference to a candidate output.
            k = 27 + col % 4
            d = 1 + (col // 4) % 8
            m[col] = ((1 << k) - d) << (31 - k)
            r[col] = k + 1
            values = []
            for sign in (-1, 1):
                for delta in range(-8, 9):
                    values.append(sign * ((1 << k) + d + delta))
            a[:, col] = np.resize(np.array(values, dtype=np.int32), rows)
    elif style == "witness":
        a.fill(1073741825); m.fill(2147483646); r.fill(31); z.fill(0)
    elif style == "input_alias":
        a[0] = m
        z[:] = r
    return a, m, r, z


class _Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(_Tensor)


class _CPU:
    int8, int32, int64, float64 = np.int8, np.int32, np.int64, np.float64

    @staticmethod
    def full(shape, value, dtype):
        # Triton typed scalars retain their type in broadcasting. A NumPy
        # zero-dimensional array can instead undergo value-based promotion.
        return np.full((1,) if shape == () else shape, value, dtype=dtype).view(_Tensor)

    @staticmethod
    def minimum(left, right):
        return np.minimum(left, right).view(_Tensor)

    @staticmethod
    def maximum(left, right):
        return np.maximum(left, right).view(_Tensor)


class _Libdevice:
    @staticmethod
    def div_rn(left, right):
        return np.divide(left, right).view(_Tensor)

    @staticmethod
    def floor(value):
        return np.floor(value).view(_Tensor)


def public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
                "_rescale_scalar", "validate_inputs", "reference", "make_inputs"}:
            node.decorator_list = []
            selected.append(node)
    namespace = {"torch": torch, "tl": _CPU(), "libdevice": _Libdevice()}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def compare(observed, values):
    target = independent_reference(*values)
    assert observed.shape == target.shape and observed.dtype == np.int8
    delta = np.abs(observed.astype(np.int16) - target.astype(np.int16))
    failed = delta != 0
    witness = None
    if failed.any():
        row, col = map(int, np.argwhere(failed)[0])
        a, m, r, z = values
        witness = {"index": [row, col], "accumulator": int(a[row, col]),
                   "multiplier": int(m[col]), "right_shift": int(r[col]),
                   "zero_point": int(z[col]), "actual": int(observed[row, col]),
                   "reference": int(target[row, col]), "allowed_error": 0}
    return {"numeric_passed": not bool(failed.any()), "elements": int(target.size),
            "definite_failures": int(failed.sum()), "max_abs_error": int(delta.max()),
            "max_abs_error_upper_bound": int(delta.max()), "witness": witness,
            "comparison": "exact Python-bigint quotient/remainder nearest rounding"}


def structural_audit(case_root):
    product_min = INT_MIN * INT_MAX
    product_max = INT_MAX * INT_MAX
    numerator_min = product_min + (1 << 30)
    numerator_max = product_max + (1 << 61)
    assert -(1 << 63) <= numerator_min <= numerator_max < (1 << 63)
    assert scalar_reference(1073741825, 2147483646, 31, 0) == 0
    assert int(np.floor(np.float64(1073741825) * np.float64(2147483646)
                        / np.float64(1 << 62) + .5)) == 1
    # Clipping before the zero point is a finite piecewise identity. Outside
    # [-255,255], either expression saturates to the same INT8 endpoint.
    clipped_checks = 0
    for z in range(-128, 128):
        for q in (-2**31, -257, -256, -255, -254, -129, -128, -1, 0,
                  1, 126, 127, 128, 254, 255, 256, 257, 2**31-1):
            assert min(127, max(-128, q + z)) == min(127, max(-128, min(255, max(-255, q)) + z))
            clipped_checks += 1
    return {"passed": True, "product_min": product_min, "product_max": product_max,
            "biased_numerator_min": numerator_min, "biased_numerator_max": numerator_max,
            "int64_overflow_possible": False, "clip_partition_checks": clipped_checks,
            "all_shifts_in_31_to_62": True,
            "scalar_correctness_argument": "The int32 operands convert to int64 exactly. Product and positive bias stay within signed int64. Arithmetic right shift by 31+r is floor division by its positive power of two, and adding half that divisor before floor division implements nearest rounding with ties toward positive infinity, including negative products. Clipping R to [-255,255] before adding z in [-128,127] preserves final INT8 saturation and prevents narrow integer overflow.",
            "address_argument": "One-dimensional launch uses index=128*program+lane and mask index<M*C. Each active index has one unique output and input cell. Channel index=index%C is in [0,C). Read-only parameters can alias inputs. Relative pointer arithmetic is unchanged by legal storage offsets. Output is newly allocated and has no input alias.",
            "scope": "Full-domain source arithmetic and indexing argument, finite integer and compiled-GPU cross-checks. This is not a formal proof of the compiler or every GPU execution.",
            "source_sha256": {name: hashlib.sha256((Path(case_root)/name/'kernel.py').read_bytes()).hexdigest() for name in EXPECTED}}


def cpu_audit(case_root):
    structural = structural_audit(case_root)
    checks, screens = [], {}
    for name, expected in EXPECTED.items():
        public = public_cpu(case_root, name)
        records = []
        for plan in plans():
            values = arrays(plan)
            tensors = tuple(torch.from_numpy(v.copy()) for v in values)
            public["validate_inputs"](*tensors, require_cuda=False)
            result = public["reference"](*tensors).numpy()
            reference_check = compare(result, values)
            assert reference_check["numeric_passed"], (name, plan, reference_check)
            actual = public["_rescale_scalar"](*(v.view(_Tensor) for v in values))
            records.append({"plan": plan, **compare(actual, values)})
        all_passed = all(row["numeric_passed"] for row in records)
        screens[name] = {"expected": expected,
                         "qualification_passed": all_passed if expected == "trust" else not all_passed,
                         "tests_total": len(records), "tests_passed": sum(r["numeric_passed"] for r in records),
                         "tests": records,
                         "scope": "Actual public scalar AST on NumPy fixed-width/FP64 adapter, not GPU execution."}
        checks.append({"case": name, "plans": len(records), "public_reference_exact": True})
    return {"passed": structural["passed"] and all(r["qualification_passed"] for r in screens.values()),
            "structural_audit": structural, "candidate_cpu_screens": screens,
            "public_reference_checks": checks, "oracle_dependency_hashes": dependency_hashes()}


def _load(case_root, name):
    module_name = "requant_compose_qualification_" + name
    spec = importlib.util.spec_from_file_location(module_name, Path(case_root) / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu = cpu_audit(case_root)
    assert cpu["passed"], "CPU qualification failed"
    cases = {}
    for name, expected in EXPECTED.items():
        module = _load(case_root, name)
        records = []
        for plan in plans():
            values = arrays(plan)
            inputs, backings = [], []
            for i, value in enumerate(values):
                offset = plan["offset"] + i
                backing = torch.full((offset + value.size + 23,), 713,
                                     dtype=torch.int32, device="cuda")
                tensor = backing[offset:offset + value.size].reshape(value.shape)
                tensor.copy_(torch.from_numpy(value).cuda())
                inputs.append(tensor); backings.append(backing)
            if plan["style"] == "input_alias":
                inputs[1] = inputs[0][0]
                inputs[3] = inputs[2]
            snapshots = [v.clone() for v in backings]
            module.validate_inputs(*inputs)
            actual = module.run(*inputs)
            torch.cuda.synchronize()
            assert type(actual) is torch.Tensor and actual.dtype == torch.int8
            assert actual.device == inputs[0].device and tuple(actual.shape) == values[0].shape
            public = module.reference(*inputs).cpu().numpy()
            assert compare(public, values)["numeric_passed"], (name, plan, "public reference")
            numeric = compare(actual.cpu().numpy(), values)
            preserved = all(torch.equal(before, after) for before, after in zip(snapshots, backings))
            fresh = all(actual.untyped_storage().data_ptr() != t.untyped_storage().data_ptr() for t in inputs)
            records.append({**plan, **numeric, "legal": True,
                            "passed": bool(numeric["numeric_passed"] and preserved and fresh),
                            "shape_dtype_valid": True, "inputs_preserved": preserved,
                            "inputs_and_backing_guards_preserved": preserved,
                            "output_fresh": fresh, "storage_offsets": [int(t.storage_offset()) for t in inputs],
                            "public_reference_exact": True})
        failed = [row for row in records if not row["passed"]]
        cases[name] = {"ground_truth": expected, "tests_total": len(records),
                       "tests_passed": len(records) - len(failed), "tests": records,
                       "counterexamples": failed,
                       "qualification_passed": bool(any(row["definite_failures"] for row in records) if expected == "reject" else not failed),
                       "label_scope": "One fixed verdict over the entire public domain; legal inputs supply evidence only.",
                       "gpu_elements_compared": sum(row["elements"] for row in records),
                       "analysis": "Exact integer oracle, full-domain source bounds and arithmetic identity, separate GPU rounding-boundary, shape, offset, alias, preservation and allocation checks."}
    return {"cases": cases, "cpu_audit": cpu,
            "qualification_passed": all(row["qualification_passed"] for row in cases.values()),
            "oracle_dependency_hashes": dependency_hashes()}
