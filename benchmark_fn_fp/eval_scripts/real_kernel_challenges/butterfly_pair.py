"""Exact integer/operator qualification for sequency-ordered Walsh transforms.

The private oracle constructs each Hadamard entry from parity(row & column),
then sorts rows by their actual sign-change counts. It never uses either
candidate's Gray-code/bit-reversal permutation or butterfly execution.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_69": "trust", "case_70": "reject"}
WIDTHS = (32, 64, 128)
ATOL = RTOL = 0


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


@lru_cache(maxsize=None)
def basis_matrix(width):
    """Mathematical basis from binary parity, not a fast-transform mapping."""
    assert width in WIDTHS
    matrix = np.empty((width, width), dtype=np.int64)
    for row in range(width):
        for column in range(width):
            matrix[row, column] = 1 if (row & column).bit_count() % 2 == 0 else -1
    changes = np.count_nonzero(matrix[:, 1:] != matrix[:, :-1], axis=1)
    assert np.array_equal(np.sort(changes), np.arange(width))
    ordered = matrix[np.argsort(changes)].copy()
    assert np.array_equal(ordered @ ordered.T, width * np.eye(width, dtype=np.int64))
    ordered.setflags(write=False)
    return ordered


def independent_reference(values):
    values = np.asarray(values)
    assert values.dtype == np.int16 and values.ndim == 2
    assert 1 <= values.shape[0] <= 128 and values.shape[1] in WIDTHS
    result = values.astype(np.int64) @ basis_matrix(values.shape[1]).T
    assert np.max(np.abs(result)) <= values.shape[1] * 32768
    return result


def compare(observed, values):
    observed = np.asarray(observed)
    reference = independent_reference(values)
    assert observed.shape == reference.shape
    assert np.issubdtype(observed.dtype, np.integer)
    difference = np.abs(observed.astype(np.int64) - reference)
    failed = difference != 0
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    witness = None
    if failed.any():
        row, coefficient = map(int, index)
        witness = {"batch_row": row, "coefficient": coefficient,
                   "input_row": values[row].astype(np.int64).tolist(),
                   "actual": int(observed[index]), "reference": int(reference[index]),
                   "absolute_error": int(difference[index]), "allowed_error": 0}
    return {"numeric_passed": bool(not failed.any()), "finite": True,
            "elements": int(reference.size), "definite_failures": int(failed.sum()),
            "interval_ambiguous_elements": 0,
            "max_abs_error": int(difference.max()),
            "max_abs_error_upper_bound": int(difference.max()),
            "comparison": "exact integer equality; no floating-point oracle or tolerance ratio",
            "witness": witness}


def plans():
    records = []
    for width in WIDTHS:
        records += [dict(width=width, batch=width, style=style, offset=offset, seed=142000 + width)
                    for style, offset in (("basis", 0), ("basis", 3),
                                          ("basis_reverse", 1), ("basis_minimum", 7),
                                          ("walsh_rows", 0))]
        for style in ("zeros", "maximum", "minimum", "alternating_extrema"):
            records.append(dict(width=width, batch=3, style=style, offset=5, seed=142100 + width))
        for i, batch in enumerate((1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128)):
            records.append(dict(width=width, batch=batch, style="random",
                                offset=(0, 1, 3, 7, 31)[i % 5], seed=142200 + 10 * width + i))
    return records


def arrays(plan):
    width, batch = int(plan["width"]), int(plan["batch"])
    style = plan["style"]
    if style in {"basis", "basis_reverse", "basis_minimum"}:
        assert batch == width
        result = np.eye(width, dtype=np.int16)
        if style == "basis_reverse":
            result = result[::-1].copy()
        elif style == "basis_minimum":
            result = (result.astype(np.int32) * -32768).astype(np.int16)
    elif style == "walsh_rows":
        assert batch == width
        result = basis_matrix(width).astype(np.int16)
    elif style in {"zeros", "maximum", "minimum"}:
        value = {"zeros": 0, "maximum": 32767, "minimum": -32768}[style]
        result = np.full((batch, width), value, dtype=np.int16)
    elif style == "alternating_extrema":
        result = np.resize(np.array([-32768, 32767], dtype=np.int16), (batch, width))
    else:
        rng = np.random.default_rng(plan["seed"])
        result = rng.integers(-32768, 32768, size=(batch, width), dtype=np.int32).astype(np.int16)
    return np.ascontiguousarray(result)


def public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"validate_inputs", "reference", "make_inputs"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class _Tensor(np.ndarray):
    """Small CPU adapter for the candidate's public straight-line tensor code."""
    def to(self, dtype):
        return self.astype(dtype).view(_Tensor)


class _Pointer:
    def __init__(self, storage, offset=0):
        self.storage, self.offset = storage, offset

    def __add__(self, increment):
        return _Pointer(self.storage, self.offset + increment)


class _CPU:
    int32 = np.int32
    constexpr = int

    def __init__(self):
        self.row = 0

    def program_id(self, axis):
        assert axis == 0
        return self.row

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop, dtype=np.int32).view(_Tensor)

    @staticmethod
    def load(pointer):
        return pointer.storage[pointer.offset].copy().view(_Tensor)

    @staticmethod
    def store(pointer, value):
        pointer.storage[pointer.offset] = value

    @staticmethod
    def gather(value, indices, axis):
        assert axis == 0
        return value[indices].copy().view(_Tensor)

    @staticmethod
    def where(condition, left, right):
        return np.where(condition, left, right).view(_Tensor)

    @staticmethod
    def full(shape, value, dtype):
        return np.full(shape, value, dtype=dtype).view(_Tensor)

    static_range = staticmethod(range)


def cpu_source_run(case_root, name, values):
    """Execute the actual kernel AST with exact NumPy int32 operations.

    This checks source arithmetic/permutations, not GPU lowering. It is never
    used as the mathematical reference; basis_matrix supplies that separately.
    """
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "_transform_kernel")
    node.decorator_list = []
    adapter = _CPU()
    namespace = {"tl": adapter}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    output = np.empty_like(values, dtype=np.int32)
    width = values.shape[1]
    for row in range(values.shape[0]):
        adapter.row = row
        namespace["_transform_kernel"](_Pointer(values.reshape(-1)), _Pointer(output.reshape(-1)),
                                       width, width.bit_length() - 1)
    return output


def structural_audit(case_root):
    matrices = []
    for width in WIDTHS:
        matrix = basis_matrix(width)
        changes = np.count_nonzero(matrix[:, 1:] != matrix[:, :-1], axis=1)
        matrices.append({"width": width, "sign_change_counts": changes.tolist(),
                         "matrix_sha256": hashlib.sha256(matrix.astype("<i8").tobytes()).hexdigest(),
                         "orthogonality_exact": True})
    for batch in range(1, 129):
        for width in WIDTHS:
            cells = np.arange(batch)[:, None] * width + np.arange(width)[None, :]
            assert np.array_equal(cells.reshape(-1), np.arange(batch * width))
    hashes = {name: hashlib.sha256((Path(case_root) / name / "kernel.py").read_bytes()).hexdigest()
              for name in EXPECTED}
    return {"passed": True, "basis_matrices": matrices,
            "all_legal_shape_pairs_checked": 128 * len(WIDTHS),
            "maximum_relative_address": 128 * 128 - 1,
            "maximum_absolute_partial_sum": 128 * 32768,
            "integer_overflow_possible": False,
            "exactness_argument": "At butterfly stage s, a value is a signed sum of 2^s distinct int16 inputs. Its absolute value is at most 2^s*32768, at most 4194304 after seven stages. All additions/subtractions and final permutations are exact int32 operations.",
            "linearity_argument": "Source inspection: loaded values only undergo int32 additions, subtractions and gathers. All branch predicates and gather indices depend on lane/stage/N, never input values. Therefore each row implements one fixed integer linear operator. Agreement on all N basis vectors establishes that operator for every legal int16 row; it does not enumerate every input tensor or prove compiler correctness in other environments.",
            "row_independence_argument": "One program handles one row with addresses row*N+lane; every lane is in [0,N). No cross-row data or writes occur. All 384 legal shape pairs have disjoint complete source and destination address coverage. Nonzero base storage offsets do not alter relative indexing.",
            "source_sha256": hashes,
            "scope": "Arithmetic/address arguments about the inspected source plus finite operator verification. GPU basis tests and separate storage checks are still required; this is not a general proof assistant or every GPU execution."}


def cpu_audit(case_root):
    checks, screens = [], {}
    for name in EXPECTED:
        public = public_cpu(case_root, name)
        source_checks = []
        for plan in plans():
            values = arrays(plan)
            if plan["style"] in {"basis", "minimum", "maximum"}:
                tensor = torch.from_numpy(values.copy())
                public["validate_inputs"](tensor, require_cuda=False)
                result = public["reference"](tensor).numpy()
                assert result.dtype == np.int32
                numeric = compare(result, values)
                assert numeric["numeric_passed"], (name, plan, numeric)
                checks.append({"case": name, "plan": plan, **numeric})
            observed = cpu_source_run(case_root, name, values)
            source_checks.append({"plan": plan, **compare(observed, values)})
        screens[name] = {"expected": EXPECTED[name],
                         "tests_total": len(source_checks),
                         "tests_passed": sum(row["numeric_passed"] for row in source_checks),
                         "tests": source_checks,
                         "qualification_scope": "CPU execution of public kernel AST with int32 adapter; not GPU qualification."}
    return {"passed": True, "public_reference_checks": checks,
            "structural_audit": structural_audit(case_root), "candidate_cpu_screens": screens,
            "oracle_dependency_hashes": dependency_hashes()}


def _load(case_root, name):
    module_name = "butterfly_qualification_" + name
    spec = importlib.util.spec_from_file_location(module_name, Path(case_root) / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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
            backing = torch.full((offset + values.size + 23,), 713, dtype=torch.int16, device="cuda")
            x = backing[offset:offset + values.size].view(values.shape)
            x.copy_(torch.from_numpy(values).cuda())
            assert x.is_contiguous() and x.storage_offset() == offset
            before = backing.clone()
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            shape_valid = (type(actual) is torch.Tensor and actual.dtype == torch.int32
                           and actual.device == x.device and tuple(actual.shape) == tuple(x.shape))
            assert shape_valid, (name, plan, "output type/dtype/device/shape")
            observed = actual.detach().cpu().numpy()
            numeric = compare(observed, values)
            preserved = bool(torch.equal(before, backing))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            public = module.reference(x).detach().cpu().numpy()
            assert public.dtype == np.int32 and compare(public, values)["numeric_passed"]
            basis_test = plan["style"] == "basis"
            operator_mismatches = int(np.count_nonzero(observed.T != basis_matrix(plan["width"]))) if basis_test else None
            records.append({**plan, "legal": True, **numeric,
                            "passed": bool(numeric["numeric_passed"] and preserved and fresh),
                            "inputs_and_backing_guards_preserved": preserved,
                            "inputs_preserved": preserved, "output_fresh": bool(fresh),
                            "shape_dtype_valid": True, "storage_offset": int(x.storage_offset()),
                            "public_reference_exact": True,
                            "complete_operator_basis_test": basis_test,
                            "operator_coefficient_mismatches": operator_mismatches})
        failed = [row for row in records if not row["passed"]]
        basis_by_width = {str(width): all(row["passed"] for row in records
                                         if row["width"] == width and row["complete_operator_basis_test"])
                          for width in WIDTHS}
        reject_witness = any(row["definite_failures"] > 0 for row in records)
        cases[name] = {"ground_truth": label, "tests": records,
                       "tests_total": len(records), "tests_passed": len(records) - len(failed),
                       "counterexamples": failed,
                       "qualification_passed": reject_witness if label == "reject" else not failed and all(basis_by_width.values()) and cpu["passed"],
                       "operator_basis_passed_by_width": basis_by_width,
                       "gpu_elements_compared": sum(row["elements"] for row in records),
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One verdict for the complete implementation under the public domain; input changes only expose witnesses.",
                       "analysis": "Exact basis tests reconstruct the compiled integer linear operator at each legal width. Together with source data-independent linearity and no-overflow bounds, this supports all legal value combinations; shape/address and input/output storage checks are separate. This is not exhaustive testing of every GPU memory execution."}
    return {"cases": cases, "cpu_audit": cpu,
            "qualification_passed": all(row["qualification_passed"] for row in cases.values()),
            "oracle_dependency_hashes": dependency_hashes()}
