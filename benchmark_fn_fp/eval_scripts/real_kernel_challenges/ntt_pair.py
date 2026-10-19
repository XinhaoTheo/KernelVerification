"""Independent exact qualification of cyclic finite-field NTT kernels.

The oracle is a direct modular DFT with Python arbitrary-precision arithmetic.
It never consumes candidate coefficient banks or candidate transform code.
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

EXPECTED = {"case_77": "trust", "case_78": "reject"}
WIDTHS = (64, 128)
PRIME = 3329
ATOL = RTOL = 0


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


@lru_cache(maxsize=None)
def basis_matrix(width):
    """Independent mathematical operator; entries are arbitrary Python integers."""
    assert width in WIDTHS
    matrix = np.array([[pow(17, (256 // width) * k * j, PRIME)
                        for j in range(width)] for k in range(width)], dtype=object)
    matrix.setflags(write=False)
    return matrix


def independent_reference(values):
    values = np.asarray(values)
    assert values.dtype == np.int32 and values.ndim == 2
    assert 1 <= values.shape[0] <= 128 and values.shape[1] in WIDTHS
    assert ((values >= 0) & (values < PRIME)).all()
    # Object-array multiplication invokes Python big-integer products and sums.
    result = values.astype(object) @ basis_matrix(values.shape[1]).T
    return np.asarray(result % PRIME, dtype=np.int32)


def compare(observed, values):
    observed = np.asarray(observed)
    reference = independent_reference(values)
    assert observed.shape == reference.shape
    assert np.issubdtype(observed.dtype, np.integer)
    difference = np.abs(observed.astype(np.int64) - reference.astype(np.int64))
    failed = difference != 0
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    witness = None
    if failed.any():
        row, coefficient = map(int, index)
        witness = {"batch_row": row, "coefficient": coefficient,
                   "input_row": values[row].tolist(), "actual": int(observed[index]),
                   "reference": int(reference[index]),
                   "absolute_error": int(difference[index]), "allowed_error": 0}
    canonical = bool(((observed >= 0) & (observed < PRIME)).all())
    return {"numeric_passed": bool(not failed.any() and canonical), "finite": True,
            "canonical_residues": canonical, "elements": int(reference.size),
            "definite_failures": int(failed.sum()), "interval_ambiguous_elements": 0,
            "max_abs_error": int(difference.max()),
            "max_abs_error_upper_bound": int(difference.max()),
            "comparison": "exact canonical modular DFT via Python integer arithmetic",
            "witness": witness}


def plans():
    result = []
    for width in WIDTHS:
        for style, offset in (("basis", 0), ("basis", 3), ("basis_scaled", 7),
                              ("basis_reverse", 1), ("character", 0)):
            result.append(dict(width=width, batch=width, style=style,
                               offset=offset, seed=150000 + width))
        for style in ("zeros", "maximum", "alternating", "constant"):
            result.append(dict(width=width, batch=3, style=style, offset=5, seed=150100 + width))
        for i, batch in enumerate((1, 2, 3, 7, 15, 31, 63, 64, 127, 128)):
            result.append(dict(width=width, batch=batch, style="random",
                               offset=(0, 1, 3, 7, 31)[i % 5], seed=150200 + width * 10 + i))
    return result


def arrays(plan):
    width, batch, style = int(plan["width"]), int(plan["batch"]), plan["style"]
    if style in {"basis", "basis_scaled", "basis_reverse"}:
        result = np.eye(width, dtype=np.int32)
        if style == "basis_scaled":
            result *= PRIME - 1
        elif style == "basis_reverse":
            result = result[::-1].copy()
    elif style == "character":
        result = np.asarray(basis_matrix(width), dtype=np.int32)
    elif style in {"zeros", "maximum", "constant"}:
        value = {"zeros": 0, "maximum": PRIME - 1, "constant": 1729}[style]
        result = np.full((batch, width), value, dtype=np.int32)
    elif style == "alternating":
        result = np.resize(np.array([0, PRIME - 1], dtype=np.int32), (batch, width))
    else:
        result = np.random.default_rng(plan["seed"]).integers(
            0, PRIME, size=(batch, width), dtype=np.int32)
    return np.ascontiguousarray(result)


class _Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(_Tensor)


class _Pointer:
    def __init__(self, storage, offset=0):
        self.storage, self.offset = storage, offset

    def __add__(self, increment):
        return _Pointer(self.storage, self.offset + increment)


class _CPU:
    int32, int64, constexpr = np.int32, np.int64, int
    static_range = staticmethod(range)

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


def public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name != "run":
            node.decorator_list = []
            selected.append(node)
    adapter = _CPU()
    namespace = {"torch": torch, "tl": adapter}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    namespace["adapter"] = adapter
    return namespace


def cpu_source_run(public, values):
    """Execute the actual candidate AST, not a separately handwritten butterfly."""
    width = values.shape[1]
    coefficients = np.array(public["_make_coefficients"](width), dtype=np.int32)
    assert ((coefficients >= 0) & (coefficients < PRIME)).all()
    output = np.empty_like(values)
    for row in range(values.shape[0]):
        public["adapter"].row = row
        public["_transform_kernel"](_Pointer(values.reshape(-1)), _Pointer(coefficients),
                                     _Pointer(output.reshape(-1)), width, width.bit_length() - 1)
    return output


def structural_audit(case_root):
    assert all(PRIME % p for p in range(2, int(PRIME ** 0.5) + 1))
    assert pow(17, 256, PRIME) == 1 and pow(17, 128, PRIME) == PRIME - 1
    assert (PRIME * 3327) % 65536 == 65535
    maximum_product = (PRIME - 1) ** 2
    maximum_correction_product = maximum_product * 3327
    maximum_reduction_numerator = maximum_product + 65535 * PRIME
    assert maximum_correction_product < 2**63
    assert maximum_reduction_numerator < 2**31
    assert maximum_reduction_numerator // 65536 < 2 * PRIME
    matrices = []
    for width in WIDTHS:
        lane = np.arange(width, dtype=np.int64)
        bits = width.bit_length() - 1
        reversed_lane = np.zeros(width, dtype=np.int64)
        for bit in range(bits):
            reversed_lane |= ((lane >> bit) & 1) << (bits - 1 - bit)
        assert np.array_equal(np.sort(reversed_lane), lane)
        table_length = 5 * ((1 << (2 * (bits // 2))) - 1) // 3
        if bits % 2:
            table_length += width // 2
        for stage in range(bits // 2):
            half = 1 << (2 * stage)
            local = lane % half
            group = (lane // (4 * half)) * (4 * half) + local
            assert int(group.min()) >= 0 and int((group + 3 * half).max()) < width
            start = 5 * (half - 1) // 3
            assert int((start + 4 * half + local).max()) < table_length
        if bits % 2:
            assert 5 * ((1 << (2 * (bits // 2))) - 1) // 3 + width // 2 == table_length
        matrix = basis_matrix(width)
        inverse = np.array([[pow(17, -(256 // width) * k * j, PRIME)
                             for j in range(width)] for k in range(width)], dtype=object)
        assert np.array_equal((matrix @ inverse) % PRIME, width * np.eye(width, dtype=object))
        matrices.append({"width": width, "root": pow(17, 256 // width, PRIME),
                         "coefficient_table_length": table_length,
                         "bit_reversal_is_permutation": True,
                         "all_gather_and_coefficient_addresses_in_bounds": True,
                         "matrix_sha256": hashlib.sha256(np.asarray(matrix, dtype="<i8").tobytes()).hexdigest(),
                         "inverse_identity_exact": True})
        for batch in range(1, 129):
            cells = np.arange(batch)[:, None] * width + np.arange(width)[None, :]
            assert np.array_equal(cells.reshape(-1), np.arange(batch * width))
    coefficient_checks = []
    for name in EXPECTED:
        public = public_cpu(case_root, name)
        for width in WIDTHS:
            bank = np.asarray(public["_make_coefficients"](width), dtype=np.int32)
            expected_length = next(item["coefficient_table_length"] for item in matrices
                                   if item["width"] == width)
            assert len(bank) == expected_length
            # For every distinct installed coefficient, exhaust all field values.
            all_values = np.arange(PRIME, dtype=np.int64)
            for coefficient in np.unique(bank):
                p = all_values * int(coefficient)
                m = (p * 3327) & 65535
                reduced = (p + m * PRIME) >> 16
                reduced = np.where(reduced >= PRIME, reduced - PRIME, reduced)
                exact = (all_values * int(coefficient) * pow(65536, -1, PRIME)) % PRIME
                assert np.array_equal(reduced, exact)
            coefficient_checks.append({"case": name, "width": width, "table_entries": len(bank),
                                       "unique_coefficients": len(np.unique(bank)),
                                       "all_3329_values_per_coefficient_exact": True})
    return {"passed": True, "prime_verified_by_trial_division": True,
            "root_order_256_verified": True, "matrices": matrices,
            "coefficient_reduction_checks": coefficient_checks,
            "all_legal_shape_pairs_checked": 128 * len(WIDTHS),
            "maximum_relative_address": 128 * 128 - 1,
            "maximum_montgomery_product": maximum_product,
            "maximum_low_word_multiply": maximum_correction_product,
            "maximum_reduction_numerator": maximum_reduction_numerator,
            "maximum_shifted_fused_sum": 6 * PRIME - 4,
            "public_reference_sum_upper_bound": 128 * (PRIME - 1) ** 2,
            "integer_overflow_possible": False,
            "exactness_argument": "Each Montgomery call consumes canonical residues. Its int64 product and low-word product are bounded; the correction makes the numerator divisible by 2^16. Reduction is congruent to a*b/R modulo3329 and lies below2q, so one conditional subtraction is sufficient. Fused signed sums plus2q and final sums plusq are nonnegative and bounded before remainder. All steps are exact.",
            "linearity_argument": "Every gather index and coefficient depends only on width/stage/lane. Modular additions, subtractions, and multiplication by fixed field constants define a linear map over F_3329; conditional canonicalization changes representatives, not the field map. Complete basis agreement establishes the map for every canonical field-valued input row, not merely sparse inputs. The output is canonical, so field equality implies exact integer equality.",
            "row_independence_argument": "One program handles one row. All input/output relative addresses are row*N+lane (input lane bit-reversed), all gathers stay in0..N-1, all outputs have unique complete coverage. Nonzero base offsets do not alter these addresses. All256 supported batch/width address layouts are checked.",
            "source_sha256": {name: hashlib.sha256((Path(case_root) / name / "kernel.py").read_bytes()).hexdigest() for name in EXPECTED},
            "scope": "Source modular arithmetic/address proof and finite operator verification, with separate compiled GPU basis/storage checks required. This does not prove every GPU execution or another compiler/environment."}


def cpu_audit(case_root):
    checks, screens = [], {}
    for name, expected in EXPECTED.items():
        public = public_cpu(case_root, name)
        source_checks = []
        for plan in plans():
            values = arrays(plan)
            if plan["style"] in {"basis", "maximum", "random"}:
                tensor = torch.from_numpy(values.copy())
                result = public["reference"](tensor).numpy()
                assert result.dtype == np.int32
                numeric = compare(result, values)
                assert numeric["numeric_passed"], (name, plan, numeric)
                checks.append({"case": name, "plan": plan, **numeric})
            actual = cpu_source_run(public, values)
            source_checks.append({"plan": plan, **compare(actual, values)})
        all_passed = all(row["numeric_passed"] for row in source_checks)
        qualified = all_passed if expected == "trust" else not all_passed
        screens[name] = {"expected": expected, "qualification_passed": qualified,
                         "tests_total": len(source_checks),
                         "tests_passed": sum(row["numeric_passed"] for row in source_checks),
                         "tests": source_checks,
                         "qualification_scope": "Actual public kernel AST on NumPy int32/int64 adapter, not GPU qualification."}
    structural = structural_audit(case_root)
    return {"passed": bool(structural["passed"] and all(x["qualification_passed"] for x in screens.values())),
            "public_reference_checks": checks, "structural_audit": structural,
            "candidate_cpu_screens": screens, "oracle_dependency_hashes": dependency_hashes()}


def _load(case_root, name):
    module_name = "ntt_qualification_" + name
    spec = importlib.util.spec_from_file_location(module_name, Path(case_root) / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu = cpu_audit(case_root)
    assert cpu["passed"], "CPU truth qualification failed"
    cases = {}
    for name, label in EXPECTED.items():
        module = _load(case_root, name)
        records = []
        for plan in plans():
            values, offset = arrays(plan), int(plan["offset"])
            backing = torch.full((offset + values.size + 23,), 713, dtype=torch.int32, device="cuda")
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
            mismatches = int(np.count_nonzero(observed.T != basis_matrix(plan["width"]))) if basis_test else None
            records.append({**plan, "legal": True, **numeric,
                            "passed": bool(numeric["numeric_passed"] and preserved and fresh),
                            "inputs_and_backing_guards_preserved": preserved,
                            "inputs_preserved": preserved, "output_fresh": bool(fresh),
                            "shape_dtype_valid": True, "storage_offset": int(x.storage_offset()),
                            "public_reference_exact": True, "complete_operator_basis_test": basis_test,
                            "operator_coefficient_mismatches": mismatches})
        failed = [row for row in records if not row["passed"]]
        basis_by_width = {str(width): all(row["passed"] for row in records
                                         if row["width"] == width and row["complete_operator_basis_test"])
                          for width in WIDTHS}
        reject_witness = any(row["definite_failures"] > 0 for row in records)
        cases[name] = {"ground_truth": label, "tests": records,
                       "tests_total": len(records), "tests_passed": len(records) - len(failed),
                       "counterexamples": failed,
                       "qualification_passed": bool(reject_witness if label == "reject" else not failed and all(basis_by_width.values()) and cpu["passed"]),
                       "operator_basis_passed_by_width": basis_by_width,
                       "gpu_elements_compared": sum(row["elements"] for row in records),
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One fixed verdict over the complete public domain; input changes expose evidence only.",
                       "analysis": "Exact GPU basis tests recover each compiled modular linear operator. Source modular-linearity and no-overflow bounds extend the operator result to all field-valued rows. Shape/address and fresh-output/input-preservation checks are separate; arbitrary GPU execution and other environments are not exhaustively proved."}
    return {"cases": cases, "cpu_audit": cpu,
            "qualification_passed": all(row["qualification_passed"] for row in cases.values()),
            "oracle_dependency_hashes": dependency_hashes()}
