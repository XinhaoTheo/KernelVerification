"""Independent exact qualification of chunked integer affine recurrences."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_85": "trust", "case_86": "reject"}
ATOL = RTOL = 0


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def independent_reference(matrix, bias, initial):
    """Sequential Python integers; no scan, tree or candidate helper is used."""
    matrix, bias, initial = map(np.asarray, (matrix, bias, initial))
    assert matrix.dtype == np.int8 and bias.dtype == initial.dtype == np.int16
    batch, length = matrix.shape[:2]
    assert matrix.shape == (batch, length, 2, 2) and bias.shape == (batch, length, 2)
    assert initial.shape == (batch, 2) and ((matrix >= -1) & (matrix <= 1)).all()
    result = np.empty((batch, length, 2), dtype=np.int64)
    for row in range(batch):
        x, y = map(int, initial[row])
        for step in range(length):
            a, b, c, d = map(int, matrix[row, step].reshape(-1))
            e, f = map(int, bias[row, step])
            x, y = a * x + b * y + e, c * x + d * y + f
            assert max(abs(x), abs(y)) < 2**48
            result[row, step] = (x, y)
    return result


def compare(observed, matrix, bias, initial):
    observed = np.asarray(observed)
    reference = independent_reference(matrix, bias, initial)
    assert observed.shape == reference.shape and observed.dtype == np.int64
    difference = np.abs(observed - reference)
    failed = observed != reference
    witness = None
    if failed.any():
        row, step, component = map(int, np.argwhere(failed)[0])
        witness = {"batch_row": row, "step": step, "component": component,
                   "matrix_prefix": matrix[row, :step + 1].tolist(),
                   "bias_prefix": bias[row, :step + 1].tolist(),
                   "initial": initial[row].tolist(),
                   "actual": int(observed[row, step, component]),
                   "reference": int(reference[row, step, component]),
                   "absolute_error": int(difference[row, step, component]), "allowed_error": 0}
    return {"numeric_passed": bool(not failed.any()), "finite": True,
            "elements": int(reference.size), "definite_failures": int(failed.sum()),
            "interval_ambiguous_elements": 0, "max_abs_error": int(difference.max()),
            "max_abs_error_upper_bound": int(difference.max()),
            "comparison": "exact sequential affine recurrence with Python arbitrary-precision integers",
            "witness": witness}


def plans():
    result = [dict(length=n, batch=1, style="random", offset=(n % 7), seed=158000 + n,
                   alias=False) for n in range(1, 33)]
    for n in (8, 9, 16, 17, 24, 25, 31, 32):
        for style in ("identity", "all_positive", "all_negative", "noncommuting", "impulses"):
            result.append(dict(length=n, batch=3, style=style, offset=5,
                               seed=158100 + n, alias=False))
    for n in (1, 7, 8, 9, 16, 17, 24, 25, 31, 32):
        for batch in (2, 7, 63, 64):
            result.append(dict(length=n, batch=batch, style="random", offset=(batch % 11),
                               seed=158200 + n * 100 + batch, alias=False))
    for n in (1, 8, 9, 17, 24, 25, 32):
        result.append(dict(length=n, batch=7, style="random", offset=7,
                           seed=158300 + n, alias=True))
    return result


def arrays(plan):
    batch, length, style = int(plan["batch"]), int(plan["length"]), plan["style"]
    rng = np.random.default_rng(plan["seed"])
    matrix = rng.integers(-1, 2, size=(batch, length, 2, 2), dtype=np.int8)
    bias = rng.integers(-32768, 32768, size=(batch, length, 2), dtype=np.int16)
    initial = rng.integers(-32768, 32768, size=(batch, 2), dtype=np.int16)
    if style == "identity":
        matrix[:] = np.eye(2, dtype=np.int8)
        bias.fill(0)
        initial[:] = (-32768, 32767)
    elif style in {"all_positive", "all_negative"}:
        matrix.fill(1 if style == "all_positive" else -1)
        bias.fill(-32768)
        initial.fill(-32768)
    elif style == "noncommuting":
        matrix[:] = np.eye(2, dtype=np.int8)
        bias.fill(0)
        initial[:] = (1, 2)
        # Left half and third chunk have shear transforms that do not commute.
        matrix[:, 0] = ((1, 1), (0, 1))
        if length > 16:
            matrix[:, 16] = ((1, 0), (1, 1))
    elif style == "impulses":
        matrix[:] = np.eye(2, dtype=np.int8)
        for i in range(length):
            if i % 3 == 0:
                matrix[:, i] = ((0, 1), (-1, 0))
            elif i % 3 == 1:
                matrix[:, i] = ((1, 1), (0, 1))
        bias.fill(0)
        bias[:, length // 2, :] = (-32768, 32767)
        initial[:] = (0, 1)
    if plan.get("alias"):
        initial = bias.reshape(-1)[:batch * 2].reshape(batch, 2)
    return tuple(np.ascontiguousarray(a) for a in (matrix, bias, initial))


class _Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(_Tensor)


class _Pointer:
    def __init__(self, storage, offset=0):
        self.storage, self.offset = storage, offset

    def __add__(self, increment):
        return _Pointer(self.storage, self.offset + increment)


class _CPU:
    int64, constexpr = np.int64, int
    static_range = staticmethod(range)

    def __init__(self):
        self.row, self.chunk = 0, 0

    def program_id(self, axis):
        return (self.row, self.chunk)[axis]

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop, dtype=np.int32).view(_Tensor)

    @staticmethod
    def load(pointer, mask=True, other=0):
        shape = np.broadcast_shapes(np.shape(pointer.offset), np.shape(mask))
        indices = np.broadcast_to(np.asarray(pointer.offset), shape)
        valid = np.broadcast_to(np.asarray(mask), shape)
        result = np.full(shape, other, dtype=pointer.storage.dtype)
        result[valid] = pointer.storage[indices[valid]]
        return result.view(_Tensor)

    @staticmethod
    def store(pointer, value, mask=True):
        shape = np.broadcast_shapes(np.shape(pointer.offset), np.shape(value), np.shape(mask))
        indices = np.broadcast_to(np.asarray(pointer.offset), shape)
        valid = np.broadcast_to(np.asarray(mask), shape)
        values = np.broadcast_to(np.asarray(value), shape)
        pointer.storage[indices[valid]] = values[valid]

    @staticmethod
    def where(condition, left, right):
        return np.asarray(np.where(condition, left, right)).view(_Tensor)

    @staticmethod
    def maximum(left, right):
        return np.asarray(np.maximum(left, right)).view(_Tensor)

    @staticmethod
    def gather(value, indices, axis):
        assert axis == 0
        return value[indices].copy().view(_Tensor)

    @staticmethod
    def associative_scan(values, axis, combine_fn):
        assert axis == 0
        result = [np.empty_like(v) for v in values]
        state = None
        for index in range(len(values[0])):
            incoming = tuple(v[index] for v in values)
            state = incoming if state is None else combine_fn(*state, *incoming)
            for target, value in zip(result, state):
                target[index] = value
        return tuple(result)


def public_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != "run":
            node.decorator_list = []
            nodes.append(node)
    adapter = _CPU()
    namespace = {"torch": torch, "tl": adapter}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    namespace["adapter"] = adapter
    return namespace


def cpu_source_run(public, matrix, bias, initial):
    batch, length = matrix.shape[:2]
    chunks = (length + 7) // 8
    padded = 1 << (chunks - 1).bit_length()
    prefix = np.empty(batch * length * 6, dtype=np.int64)
    summary = np.empty(batch * padded * 6, dtype=np.int64)
    carry = np.empty(batch * padded * 2, dtype=np.int64)
    output = np.empty(batch * length * 2, dtype=np.int64)
    for row in range(batch):
        public["adapter"].row = row
        for chunk in range(chunks):
            public["adapter"].chunk = chunk
            public["_local_scan"](_Pointer(matrix.reshape(-1)), _Pointer(bias.reshape(-1)),
                                    _Pointer(prefix), _Pointer(summary), length, padded, 8)
        public["_block_carries"](_Pointer(summary), _Pointer(initial.reshape(-1)), _Pointer(carry),
                                    chunks, padded, padded.bit_length() - 1)
        for chunk in range(chunks):
            public["adapter"].chunk = chunk
            public["_apply_prefix"](_Pointer(prefix), _Pointer(carry), _Pointer(output), length, padded, 8)
    return output.reshape(batch, length, 2)


def _payload_matrix(payload):
    a, b, c, d, e, f = map(int, payload)
    return np.asarray(((a, b, e), (c, d, f), (0, 0, 1)), dtype=object)


def structural_audit(case_root):
    control = public_cpu(case_root, "case_85")
    # Six payload basis vectors and structured maps check the composition terms.
    payloads = [(1, 0, 0, 1, 0, 0), (1, 1, 0, 1, 0, 0), (1, 0, 1, 1, 0, 0),
                (0, 1, -1, 0, 0, 0), (0, 0, 0, 0, 1, -1), (1, 0, 0, 1, -32768, 32767)]
    payloads += [tuple(int(j == i) for j in range(6)) for i in range(6)]
    checks = 0
    for left, right in itertools.product(payloads, repeat=2):
        actual = _payload_matrix(control["_compose"](*left, *right))
        expected = _payload_matrix(right) @ _payload_matrix(left)
        assert np.array_equal(actual, expected)
        checks += 1
    tree_records = []
    for name in EXPECTED:
        public = public_cpu(case_root, name)
        for size in (1, 2, 4):
            mismatches, total = 0, 0
            for combo in itertools.product(payloads[:4], repeat=size):
                inputs = tuple(np.asarray([x[i] for x in combo], dtype=np.int64).view(_Tensor) for i in range(6))
                output = public["_exclusive_chunks"](*inputs, size, size.bit_length() - 1)
                prefix = np.eye(3, dtype=object)
                for lane in range(size):
                    mismatches += not np.array_equal(_payload_matrix(tuple(v[lane] for v in output)), prefix)
                    prefix = _payload_matrix(combo[lane]) @ prefix
                    total += 1
            tree_records.append({"case": name, "chunks": size, "prefixes_checked": total,
                                 "mismatches": int(mismatches)})
            if name == "case_85":
                assert mismatches == 0
    for batch in range(1, 65):
        for length in range(1, 33):
            chunks = (length + 7) // 8
            actual = [row * length + chunk * 8 + lane for row in range(batch)
                      for chunk in range(chunks) for lane in range(8) if chunk * 8 + lane < length]
            assert actual == list(range(batch * length))
    return {"passed": True, "affine_composition_checks": checks, "exclusive_tree_checks": tree_records,
            "legal_shape_address_checks": 64 * 32,
            "state_magnitude_upper_bound": 32768 * (2**33 - 1),
            "matrix_partial_entry_upper_bound": 2**31,
            "bias_partial_magnitude_upper_bound": 32768 * (2**32 - 1),
            "source_correctness_argument": "Payload multiplication is multiplication of homogeneous3x3 affine matrices. The combine operation returns the right map after the left map. The local scan preserves temporal order. The upsweep holds each interval total at its right endpoint; the downsweep sends the incoming prefix to the left child and left-total-after-prefix to the right child. Identity padding is neutral. Final local maps act on each chunk's incoming state. Induction on the tree gives all sequential states.",
            "integer_bound_argument": "For t steps, matrix entries are at most2^(t-1) in magnitude and affine bias components at most32768*(2^t-1). Actual state magnitude is at most32768*(2^(t+1)-1). With t<=32 all are below2^48. In the four-lane upsweep, even discarded lane calculations compose at most32 steps; the downsweep composes at most24. Thus masked intermediate calculations also fit int64. All pointer offsets fit int32.",
            "storage_argument": "Every kernel only reads the public inputs. The first stage writes valid local prefixes and one summary per chunk. Same-stream launches order scratch producers before consumers. The carry stage masks unused summary lanes to identity; the final stage masks the tail. The host allocates distinct output storage. Pointer-relative addressing handles legal offsets and shared input storage.",
            "scope": "Source algebra and integer/address bounds establish the control for the full public value domain. Enumerated operators and compiled GPU checks test those arguments; finite samples alone are not a proof over every input or GPU environment.",
            "source_sha256": {name: hashlib.sha256((Path(case_root) / name / "kernel.py").read_bytes()).hexdigest() for name in EXPECTED}}


def cpu_audit(case_root):
    structural = structural_audit(case_root)
    cases, references = {}, []
    for name, label in EXPECTED.items():
        public = public_cpu(case_root, name)
        records = []
        for plan in plans():
            matrix, bias, initial = arrays(plan)
            actual = cpu_source_run(public, matrix, bias, initial)
            numeric = compare(actual, matrix, bias, initial)
            records.append({**plan, **numeric})
            expected = public["reference"](*(torch.from_numpy(x) for x in (matrix, bias, initial))).numpy()
            assert compare(expected, matrix, bias, initial)["numeric_passed"]
            references.append({"case": name, **plan, "passed": True})
        failures = [record for record in records if not record["numeric_passed"]]
        cases[name] = {"ground_truth": label, "tests_total": len(records),
                       "tests_passed": len(records) - len(failures), "tests": records,
                       "qualification_passed": bool(failures) if label == "reject" else not failures}
    return {"passed": all(case["qualification_passed"] for case in cases.values()),
            "candidate_cpu_screens": cases, "public_reference_checks": references,
            "structural_audit": structural, "oracle_dependency_hashes": dependency_hashes()}


def _load(case_root, name):
    module_name = "affine_scan_qualification_" + name
    spec = importlib.util.spec_from_file_location(module_name, Path(case_root) / name / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _gpu_inputs(values, offset, alias):
    matrix, bias, initial = values
    tensors, backings = [], []
    for index, array in enumerate((matrix, bias, initial)):
        if index == 2 and alias:
            tensor = tensors[1].reshape(-1)[:initial.size].view(initial.shape)
            assert torch.equal(tensor.cpu(), torch.from_numpy(initial))
            tensors.append(tensor)
            continue
        dtype = (torch.int8, torch.int16, torch.int16)[index]
        backing = torch.full((offset + array.size + 19,), 7, dtype=dtype, device="cuda")
        tensor = backing[offset:offset + array.size].view(array.shape)
        tensor.copy_(torch.from_numpy(array).cuda())
        backings.append(backing)
        tensors.append(tensor)
    return tensors, backings


def validate(case_root):
    cpu = cpu_audit(case_root)
    assert cpu["passed"], "CPU truth qualification failed"
    cases = {}
    for name, label in EXPECTED.items():
        module, records = _load(case_root, name), []
        for plan in plans():
            values = arrays(plan)
            tensors, backings = _gpu_inputs(values, int(plan["offset"]), bool(plan["alias"]))
            before = [backing.clone() for backing in backings]
            module.validate_inputs(*tensors)
            actual = module.run(*tensors)
            torch.cuda.synchronize()
            batch, length = values[0].shape[:2]
            assert type(actual) is torch.Tensor and actual.dtype == torch.int64
            assert actual.device == tensors[0].device and tuple(actual.shape) == (batch, length, 2)
            numeric = compare(actual.detach().cpu().numpy(), *values)
            public = module.reference(*tensors).detach().cpu().numpy()
            assert compare(public, *values)["numeric_passed"]
            preserved = all(bool(torch.equal(old, new)) for old, new in zip(before, backings))
            fresh = all(actual.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
                        for tensor in tensors)
            alias_observed = tensors[1].untyped_storage().data_ptr() == tensors[2].untyped_storage().data_ptr()
            assert alias_observed == bool(plan["alias"])
            records.append({**plan, "legal": True, **numeric,
                            "passed": bool(numeric["numeric_passed"] and preserved and fresh),
                            "inputs_and_backing_guards_preserved": preserved,
                            "inputs_preserved": preserved, "output_fresh": fresh,
                            "shape_dtype_valid": True, "storage_offsets": [int(x.storage_offset()) for x in tensors],
                            "bias_initial_alias_observed": alias_observed,
                            "public_reference_exact": True})
        failed = [record for record in records if not record["passed"]]
        cases[name] = {"ground_truth": label, "tests": records, "tests_total": len(records),
                       "tests_passed": len(records) - len(failed), "counterexamples": failed,
                       "qualification_passed": bool(any(record["definite_failures"] for record in records))
                       if label == "reject" else not failed,
                       "gpu_elements_compared": sum(record["elements"] for record in records),
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One fixed verdict over every public matrix, bias and initial value; inputs provide evidence only.",
                       "analysis": "Exact sequential Python-int oracle plus source affine-tree proof and integer bounds. GPU checks include every legal sequence length, chunk boundaries, noncommuting shears, signed extremes, offsets, shared input storage, backing guards and fresh output. No claim exhausts all input combinations or GPU environments."}
    return {"cases": cases, "cpu_audit": cpu,
            "qualification_passed": all(item["qualification_passed"] for item in cases.values()),
            "oracle_dependency_hashes": dependency_hashes()}
