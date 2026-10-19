"""Independent exact bitwise oracle and compiled GPU qualification for CRC rows."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import zlib

import numpy as np
import torch

EXPECTED = {"case_79": "trust", "case_80": "reject"}
WIDTHS = (64, 128)
POLY = 0xEDB88320
MASK = 0xFFFFFFFF
ATOL = RTOL = 0


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def independent_reference(values):
    """Direct shift-register recurrence, without candidate tables or factors."""
    values = np.asarray(values)
    assert values.dtype == np.uint8 and values.ndim == 2
    state = np.full(values.shape[0], MASK, dtype=np.uint64)
    for index in range(values.shape[1]):
        state ^= values[:, index].astype(np.uint64)
        for _ in range(8):
            state = (state >> 1) ^ ((state & 1) * np.uint64(POLY))
    return (state ^ np.uint64(MASK)).astype(np.int64)


def compare(observed, values):
    observed = np.asarray(observed)
    reference = independent_reference(values)
    assert observed.shape == reference.shape and np.issubdtype(observed.dtype, np.integer)
    differences = np.abs(observed.astype(np.int64) - reference)
    failed = observed != reference
    index = int(np.argmax(differences))
    witness = None
    if failed.any():
        witness = {"batch_row": index, "input_row": values[index].tolist(),
                   "actual": int(observed[index]), "reference": int(reference[index]),
                   "actual_hex": hex(int(observed[index])), "reference_hex": hex(int(reference[index])),
                   "absolute_error": int(differences[index]), "allowed_error": 0}
    canonical = bool(((observed >= 0) & (observed <= MASK)).all())
    return {"numeric_passed": bool(not failed.any() and canonical), "finite": True,
            "canonical_unsigned_checksums": canonical, "elements": int(reference.size),
            "definite_failures": int(failed.sum()), "interval_ambiguous_elements": 0,
            "max_abs_error": int(differences.max()), "max_abs_error_upper_bound": int(differences.max()),
            "comparison": "exact unsigned CRC-32 from independent sequential bit recurrence", "witness": witness}


def plans():
    result = []
    for width in WIDTHS:
        for start in range(0, width * 8, 128):
            result.append(dict(width=width, batch=128, style="basis", start=start,
                               offset=(0, 1, 3, 7)[(start // 128) % 4], seed=0))
        for style, offset in (("zeros", 0), ("zeros", 17), ("maximum", 5),
                              ("alternating", 3), ("increasing", 7)):
            result.append(dict(width=width, batch=3, style=style, offset=offset, seed=0))
        for index, batch in enumerate((1, 2, 7, 31, 63, 127, 128)):
            result.append(dict(width=width, batch=batch, style="random", offset=(0, 1, 3, 7, 31)[index % 5],
                               seed=152000 + width * 10 + index))
    return result


def arrays(plan):
    width, batch, style = int(plan["width"]), int(plan["batch"]), plan["style"]
    if style == "basis":
        result = np.zeros((batch, width), dtype=np.uint8)
        bit = np.arange(plan["start"], plan["start"] + batch)
        result[np.arange(batch), bit // 8] = (1 << (bit % 8)).astype(np.uint8)
    elif style in {"zeros", "maximum"}:
        result = np.full((batch, width), 0 if style == "zeros" else 255, dtype=np.uint8)
    elif style == "alternating":
        result = np.resize(np.array([0x55, 0xAA], dtype=np.uint8), (batch, width))
    elif style == "increasing":
        result = np.arange(batch * width, dtype=np.int64).reshape(batch, width).astype(np.uint8)
    else:
        result = np.random.default_rng(plan["seed"]).integers(0, 256, (batch, width), dtype=np.uint8)
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
    uint32, int64, constexpr = np.uint32, np.int64, int
    static_range = staticmethod(range)

    def __init__(self):
        self.row = 0

    def program_id(self, axis):
        assert axis == 0
        return self.row

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop, dtype=np.int64).view(_Tensor)

    @staticmethod
    def load(pointer):
        return np.asarray(pointer.storage[pointer.offset]).copy().view(_Tensor)

    @staticmethod
    def store(pointer, value):
        pointer.storage[pointer.offset] = value

    @staticmethod
    def where(condition, left, right):
        return np.where(condition, left, right).view(_Tensor)

    @staticmethod
    def full(shape, value, dtype):
        return np.full(shape, value, dtype=dtype).view(_Tensor)

    @staticmethod
    def reduce(value, axis, combine_fn):
        assert axis == 0 and combine_fn.__name__ == "_xor"
        return np.asarray(np.bitwise_xor.reduce(value, axis=axis)).view(_Tensor)


def public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name != "run":
            node.decorator_list = []
            nodes.append(node)
    adapter = _CPU()
    namespace = {"torch": torch, "zlib": zlib, "tl": adapter}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["adapter"] = adapter
    return namespace


def cpu_source_run(public, values):
    """Execute the actual public kernel functions using exact NumPy bit operations."""
    width = values.shape[1]
    coefficients = np.asarray(public["_make_coefficients"](width), dtype=np.int64)
    output = np.empty(values.shape[0], dtype=np.int64)
    for row in range(values.shape[0]):
        public["adapter"].row = row
        public["_checksum_kernel"](_Pointer(values.reshape(-1)), _Pointer(coefficients),
                                    _Pointer(output), width, width // 4)
    return output


def structural_audit(case_root):
    known = np.frombuffer(b"123456789", dtype=np.uint8)[None, :]
    assert int(independent_reference(known)[0]) == 0xCBF43926
    operator_checks = []
    for width in WIDTHS:
        for batch in range(1, 129):
            indices = (np.arange(batch)[:, None, None] * width
                       + np.arange(width // 4)[None, :, None] * 4 + np.arange(4)[None, None, :])
            assert np.array_equal(indices.reshape(-1), np.arange(batch * width))
        for name in EXPECTED:
            public = public_cpu(case_root, name)
            factors = public["_make_coefficients"](width)
            assert len(factors) == width // 4 + 1 and all(0 <= x <= MASK for x in factors)
            # Certify each installed polynomial action on the complete 32-bit basis.
            # Each action is linear in its right operand because left is constant.
            bad = 0
            for chunk, factor in enumerate(factors[:-1]):
                remaining_bytes = width - 4 * (chunk + 1)
                for bit in range(32):
                    state = 1 << bit
                    expected = state
                    for _ in range(remaining_bytes * 8):
                        expected = (expected >> 1) ^ (POLY if expected & 1 else 0)
                    observed = public["_polynomial_product"](factor, state)
                    bad += observed != expected
            operator_checks.append({"case": name, "width": width,
                                    "factor_bit_basis_checks": width // 4 * 32,
                                    "factor_bit_basis_mismatches": int(bad)})
    return {"passed": True, "known_crc_check_123456789": "0xcbf43926",
            "factor_operator_checks": operator_checks,
            "all_legal_batch_width_address_layouts_checked": 256,
            "maximum_relative_input_address": 128 * 128 - 1,
            "maximum_relative_factor_address": 32,
            "integer_overflow_possible": False,
            "integer_argument": "All polynomial states and coefficients are unsigned 32-bit values. Right shifts and XOR preserve that range. Host Python integers are unbounded. The final uint32 value converts exactly to nonnegative int64. All row and chunk addresses fit int32; each input byte is read once and each output row is stored once.",
            "affine_argument": "Chunk updates start from zero. Bit shifts and the conditional XOR by the polynomial are linear over GF(2). Polynomial multiplication uses a fixed left factor, so it is linear in the chunk state. The reduction is XOR. The correction depends only on width. Thus the public candidate is an affine map from 8*N input bits to 32 output bits. Zero input and all 8*N unit-bit inputs determine this map completely. Candidate and oracle agreement on this affine basis extends to every allowed byte sequence, with separate compiled GPU checks required.",
            "row_and_storage_argument": "One program handles one row. The address enumeration covers all 256 legal batch/width pairs, with no overlaps or omissions. Pointer-relative addressing supports all legal base offsets. Coefficient indices lie in the allocated table. Fresh allocation and input preservation are checked separately.",
            "source_sha256": {name: hashlib.sha256((Path(case_root) / name / "kernel.py").read_bytes()).hexdigest() for name in EXPECTED},
            "scope": "Exact source affine-map/address argument plus finite operator checks. Compiled GPU affine bases and storage checks remain required; no claim covers all GPU environments or hardware faults."}


def cpu_audit(case_root):
    structural = structural_audit(case_root)
    cases, reference_checks = {}, []
    for name, label in EXPECTED.items():
        public = public_cpu(case_root, name)
        tests = []
        for plan in plans():
            values = arrays(plan)
            observed = cpu_source_run(public, values)
            tests.append({**plan, **compare(observed, values)})
            reference = public["reference"](torch.from_numpy(values.copy())).numpy()
            result = compare(reference, values)
            assert result["numeric_passed"] and reference.dtype == np.int64
            reference_checks.append({"case": name, **plan, "passed": True})
        passed = all(row["numeric_passed"] for row in tests)
        cases[name] = {"ground_truth": label,
                       "qualification_passed": bool(passed if label == "trust" else not passed),
                       "tests_total": len(tests), "tests_passed": sum(row["numeric_passed"] for row in tests),
                       "tests": tests}
    assert next(x for x in structural["factor_operator_checks"] if x["case"] == "case_79")["factor_bit_basis_mismatches"] == 0
    return {"passed": all(item["qualification_passed"] for item in cases.values()),
            "candidate_cpu_screens": cases, "public_reference_checks": reference_checks,
            "structural_audit": structural, "oracle_dependency_hashes": dependency_hashes()}


def _load(case_root, name):
    module_name = "crc_qualification_" + name
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
        module, records = _load(case_root, name), []
        for plan in plans():
            values, offset = arrays(plan), int(plan["offset"])
            backing = torch.full((offset + values.size + 23,), 173, dtype=torch.uint8, device="cuda")
            x = backing[offset:offset + values.size].view(values.shape)
            x.copy_(torch.from_numpy(values).cuda())
            before = backing.clone()
            module.validate_inputs(x)
            actual = module.run(x)
            torch.cuda.synchronize()
            assert type(actual) is torch.Tensor and actual.dtype == torch.int64
            assert actual.device == x.device and tuple(actual.shape) == (values.shape[0],)
            observed = actual.detach().cpu().numpy()
            numeric = compare(observed, values)
            public = module.reference(x).detach().cpu().numpy()
            assert public.dtype == np.int64 and compare(public, values)["numeric_passed"]
            preserved = bool(torch.equal(backing, before))
            fresh = actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            records.append({**plan, "legal": True, **numeric,
                            "passed": bool(numeric["numeric_passed"] and preserved and fresh),
                            "inputs_and_backing_guards_preserved": preserved,
                            "inputs_preserved": preserved, "output_fresh": bool(fresh),
                            "shape_dtype_valid": True, "storage_offset": int(x.storage_offset()),
                            "public_reference_exact": True,
                            "complete_affine_basis_component": plan["style"] in {"basis", "zeros"}})
        failed = [item for item in records if not item["passed"]]
        affine = {str(width): all(item["passed"] for item in records
                                  if item["width"] == width and item["complete_affine_basis_component"])
                  for width in WIDTHS}
        cases[name] = {"ground_truth": label, "tests": records, "tests_total": len(records),
                       "tests_passed": len(records) - len(failed), "counterexamples": failed,
                       "qualification_passed": bool(any(item["definite_failures"] for item in records)
                                                    if label == "reject" else not failed and all(affine.values())),
                       "complete_affine_basis_passed_by_width": affine,
                       "gpu_elements_compared": sum(item["elements"] for item in records),
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One fixed verdict over every allowed byte array; inputs provide evidence only.",
                       "analysis": "Zero rows and every input-bit basis vector recover the compiled affine checksum operator at both widths. The source affine/address/no-overflow argument extends the value result to all allowed byte arrays. This does not exhaust every tensor execution or GPU environment."}
    return {"cases": cases, "cpu_audit": cpu,
            "qualification_passed": all(item["qualification_passed"] for item in cases.values()),
            "oracle_dependency_hashes": dependency_hashes()}
