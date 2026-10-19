"""Independent direct-DFT qualification for packed real FFT cases162/163."""
from __future__ import annotations

import ast
from fractions import Fraction
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

import gmpy2
import numpy as np
import torch

EXPECTED = {"case_89": "trust", "case_90": "reject"}
WIDTHS = (64, 128, 256)
ATOL, RTOL = 1e-4, 1e-6


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


@lru_cache(maxsize=None)
def direct_coefficients(width):
    """Direct mathematical DFT coefficients, independent of packed FFT tables."""
    assert width in WIDTHS
    matrix = np.empty((width // 2 + 1, width, 2), dtype=np.float64)
    radii = np.empty_like(matrix)
    with gmpy2.context(precision=256, round=gmpy2.RoundToNearest):
        pi = gmpy2.const_pi()
        high_precision_error = gmpy2.mpfr(2) ** -240
        roots = []
        for residue in range(width):
            if (4 * residue) % width == 0:
                roots.append([(1, 0), (0, -1), (-1, 0), (0, 1)][(4 * residue // width) % 4])
            else:
                angle = 2 * pi * residue / width
                roots.append((gmpy2.cos(angle), -gmpy2.sin(angle)))
        for k in range(width // 2 + 1):
            for j in range(width):
                for c, value in enumerate(roots[(j * k) % width]):
                    rounded = float(value)
                    error = abs(gmpy2.mpfr(rounded) - value) + high_precision_error
                    matrix[k, j, c] = rounded
                    radii[k, j, c] = np.nextafter(float(error), np.inf)
    return matrix, radii


def independent_reference(values):
    """FP64 direct matrix product with an explicit conservative error interval."""
    values = np.asarray(values)
    assert values.dtype == np.float16 and values.ndim == 2
    assert values.shape[1] in WIDTHS and np.isfinite(values).all()
    width = values.shape[1]
    coefficients, coefficient_radius = direct_coefficients(width)
    x = values.astype(np.float64)
    center = np.stack([x @ coefficients[:, :, c].T for c in range(2)], axis=-1)
    # At most N multiplications and N additions per component, regardless of
    # whether BLAS uses separate operations, reassociation or fused operations.
    u = 2.0 ** -53
    gamma = (2 * width * u) / (1 - 2 * width * u)
    magnitude = np.abs(x).sum(axis=1)[:, None, None]
    root_bound = float(coefficient_radius.max())
    # Use twice the analytical bound to cover computation of this envelope.
    radius = np.broadcast_to(2 * (gamma + root_bound) * magnitude + 2.0 ** -220, center.shape).copy()
    return center, radius


def compare(observed, values):
    center, radius = independent_reference(values)
    observed = np.asarray(observed, dtype=np.float64)
    assert observed.shape == center.shape
    finite = np.isfinite(observed)
    distance = np.abs(observed - center)
    distance_lower = np.maximum(0.0, np.nextafter(distance, -np.inf))
    distance_upper = np.nextafter(distance, np.inf)
    error_upper = np.nextafter(distance_upper + radius, np.inf)
    error_lower = np.maximum(0.0, np.nextafter(distance_lower - radius, -np.inf))
    allowed_lower = (ATOL + RTOL * np.maximum(0.0, np.abs(center) - radius)) * (1 - 8 * np.finfo(np.float64).eps)
    allowed_upper = (ATOL + RTOL * (np.abs(center) + radius)) * (1 + 8 * np.finfo(np.float64).eps)
    passed = finite & (error_upper <= allowed_lower)
    failed = ~finite | (error_lower > allowed_upper)
    ambiguous = ~passed & ~failed
    index = np.unravel_index(int(np.argmax(error_upper / allowed_lower)), center.shape)
    witness = None
    if failed.any():
        index = tuple(int(i) for i in np.argwhere(failed)[0])
        row, frequency, component = index
        witness = {"batch_row": row, "frequency": frequency, "component": "real" if component == 0 else "imaginary",
                   "input_row": np.asarray(values)[row].astype(float).tolist(), "actual": float(observed[index]),
                   "reference_center": float(center[index]), "reference_radius": float(radius[index]),
                   "error_lower_bound": float(error_lower[index]), "allowed_error_upper_bound": float(allowed_upper[index])}
    return {"numeric_passed": bool(passed.all()), "finite": bool(finite.all()), "elements": int(center.size),
            "definite_failures": int(failed.sum()), "interval_ambiguous_elements": int(ambiguous.sum()),
            "max_abs_error": float(distance.max()), "max_abs_error_upper_bound": float(error_upper.max()),
            "max_error_tolerance_ratio": float((error_upper / allowed_lower).max()),
            "max_reference_radius": float(radius.max()), "witness": witness}


def arithmetic_audit():
    u = Fraction(1, 1 << 53)
    u32 = Fraction(1, 1 << 24)
    magnitude, error = Fraction(65504), Fraction(0)
    steps = []
    for stage in range(7):
        error = 4 * error + 32 * u * magnitude
        magnitude *= 4
        assert error < magnitude
        steps.append({"stage": stage + 1, "ideal_component_magnitude_bound": float(magnitude), "error_bound": float(error)})
    reconstruction_error = 3 * error + 32 * u * magnitude
    final_absolute = (1 + u32) * reconstruction_error + Fraction(1, 1 << 126)
    assert final_absolute < Fraction(1, 10000) and u32 < Fraction(1, 1000000)
    return {"passed": True, "stage_bounds": steps, "pre_output_absolute_error_bound": float(reconstruction_error),
            "final_absolute_error_term": float(final_absolute), "final_relative_error_term": float(u32),
            "proof": [
                "The mathematical half-length radix-2 DIF transform is followed by the standard even/odd real-FFT reconstruction. DIF output index reverse(k) contains complex coefficient k. The reconstruction needs the conjugate of coefficient (-k) mod M, so the partner address is reverse((-k) mod M).",
                "All finite binary16 inputs convert exactly to binary64. Each stored sine or cosine has magnitude at most1 and absolute error at mostu=2^-53, certified independently against MPFR roots.",
                "Let R bound each exact intermediate component and E bound its absolute error. InitiallyR=65504,E=0. One butterfly gives newR=4R and newE=4E+32uR. This bounds two input errors, a rounded difference, both coefficient errors, two rounded products and a rounded sum. The sum branch has a smaller bound. E<R holds at every stage, so computed component magnitudes stay below2R.",
                "There are at most7complex stages. Reconstruction forms a sum and difference of two coefficients, multiplies the difference by one root, adds, then divides by2. Its component error is at most3E+32uR. DC and Nyquist use simple sums and differences with smaller bounds.",
                "All nonzero table values are multiples of2^-58. With at most8root multiplications and one exact factor1/2, nonzero intermediates are bounded below by2^-489; magnitudes are below2^35. Thus no FP64 underflow or overflow occurs. The FP32 store error is at mostu32*abs(value)+2^-126, allowing for a possible flush of a subnormal FP32 result. The resulting absolute and relative bounds are below the public1e-4 and1e-6tolerances.",
                "This argument covers every finite FP16 row. Complete impulse bases check frequency indexing and the implemented transform on those inputs at each supported width. Floating-point execution is not literally linear, so basis tests alone do not prove all-input accuracy. The roundoff argument supplies that coverage; GPU tests separately check lowering and tensor behavior."
            ]}


class _Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(_Tensor)


class _Pointer:
    def __init__(self, array, offset=0):
        self.array, self.offset = array, offset

    def __add__(self, offset):
        return _Pointer(self.array, self.offset + offset)


class _TL:
    constexpr = object
    int32 = np.int32
    float32 = np.float32
    float64 = np.float64
    row = 0
    static_range = range

    @staticmethod
    def program_id(axis):
        return _TL.row

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop, dtype=np.int32).view(_Tensor)

    @staticmethod
    def full(shape, value, dtype):
        return np.full(shape, value, dtype=dtype).view(_Tensor)

    @staticmethod
    def load(pointer):
        return np.asarray(pointer.array[pointer.offset]).view(_Tensor)

    @staticmethod
    def store(pointer, value, mask=None):
        offset = np.asarray(pointer.offset)
        value = np.broadcast_to(value, offset.shape)
        if mask is not None:
            mask = np.asarray(mask)
            if offset.ndim == 0:
                if mask.any():
                    pointer.array[int(offset)] = np.asarray(value)[mask][0] if value.ndim else value
                return
            pointer.array[offset[mask]] = value[mask]
        else:
            pointer.array[offset] = value

    @staticmethod
    def gather(values, index, axis):
        return np.take(values, index, axis=axis).view(_Tensor)

    @staticmethod
    def where(condition, left, right):
        return np.where(condition, left, right).view(_Tensor)

    minimum = staticmethod(np.minimum)
    abs = staticmethod(np.abs)


def _source_namespace(path):
    tree = ast.parse(Path(path).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_COSINE" for t in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"_reverse", "_twiddle", "_realfft", "validate_inputs", "reference", "make_inputs"}:
            node.decorator_list = []
            selected.append(node)
    namespace = {"tl": _TL, "torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def cpu_candidate(namespace, values):
    batch, n = values.shape
    m = n // 2
    output = np.full((batch, m + 1, 2), np.nan, dtype=np.float32)
    table = np.array(namespace["_COSINE"], dtype=np.float64)
    for row in range(batch):
        _TL.row = row
        namespace["_realfft"](_Pointer(values.ravel()), _Pointer(table), _Pointer(output.ravel()), n, m, m.bit_length() - 1)
    return output


def _source_binding(case_root):
    result = {}
    for case in EXPECTED:
        path = Path(case_root) / case / "kernel.py"
        tree = ast.parse(path.read_text())
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_realfft")
        assignments = {n.targets[0].id: ast.unparse(n.value) for n in kernel.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        expected = "_reverse(-k & M - 1, LOG_M)" if case == "case_89" else "-address & M - 1"
        assert assignments["partner"] == expected, assignments["partner"]
        source = path.read_text()
        assert "enable_fp_fusion=False" in source
        assert source.count(".to(tl.float64)") == 2
        result[case] = {"kernel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "partner_address": assignments["partner"], "implicit_fma_disabled": True}
    control = (Path(case_root) / "case_89" / "kernel.py").read_text()
    defect = (Path(case_root) / "case_90" / "kernel.py").read_text()
    assert control.replace("_reverse((-k) & (M - 1), LOG_M)", "(-address) & (M - 1)") == defect
    assert (Path(case_root) / "case_89" / "problem.txt").read_bytes() == (Path(case_root) / "case_90" / "problem.txt").read_bytes()
    return result


def cpu_audit(case_root):
    proof, source = arithmetic_audit(), _source_binding(case_root)
    cases = {}
    roots_audit = None
    for case, label in EXPECTED.items():
        namespace = _source_namespace(Path(case_root) / case / "kernel.py")
        with gmpy2.context(precision=256):
            errors = []
            for j, value in enumerate(namespace["_COSINE"]):
                exact = gmpy2.cos(gmpy2.const_pi() * j / 128) if j != 64 else gmpy2.mpfr(0)
                error = abs(gmpy2.mpfr(value) - exact) + gmpy2.mpfr(2) ** -240
                assert error < gmpy2.mpfr(2) ** -53
                assert abs(value) <= 1
                assert float(value) * 2**58 == int(float(value) * 2**58)
                errors.append(float(error))
            roots_audit = {"coefficients": 65, "max_absolute_error_upper_bound": max(errors), "mpfr_precision": 256, "high_precision_error_budget": float(2.0**-240), "passed": True}
        rows = []
        for width in WIDTHS:
            values = np.eye(width, dtype=np.float16)
            actual = cpu_candidate(namespace, values)
            result = compare(actual, values)
            result.update(width=width, style="complete_impulse_basis", rows=width)
            if label == "trust":
                assert result["numeric_passed"], (case, width, result)
            else:
                assert result["definite_failures"] > 0
            public = namespace["reference"](torch.from_numpy(values)).numpy()
            public_result = compare(public, values)
            assert public_result["numeric_passed"]
            result["public_reference_within_contract"] = True
            rows.append(result)
        cases[case] = {"expected_label": label, "basis_checks": rows, "passed": True}
    return {"passed": True, "arithmetic": proof, "source_binding": source, "root_table_audit": roots_audit,
            "cases": cases, "oracle": "Independent direct DFT matrix, with256-bit MPFR roots and a conservative FP64 matrix-product error envelope.",
            "scope": "Complete impulse bases check indexing and transform behavior on those inputs. Floating-point execution is not literally linear. The mathematical transform identity and roundoff bound cover all legal FP16 inputs for the control.",
            "oracle_dependency_hashes": dependency_hashes()}


def plans():
    result = []
    for width in WIDTHS:
        for style in ("basis", "basis_scaled", "zeros", "maximum", "alternating", "subnormal", "impulses"):
            batch = width if style.startswith("basis") else 7
            result.append(dict(width=width, batch=batch, style=style, offset=3 if style == "basis" else 7, seed=162000+width))
        for i, batch in enumerate((1, 3, 31, 127, 256)):
            result.append(dict(width=width, batch=batch, style="random_bits", offset=(0, 1, 7, 31, 3)[i], seed=162100+width*10+i))
    return result


def arrays(plan):
    width, batch, style = plan["width"], plan["batch"], plan["style"]
    if style.startswith("basis"):
        values = np.eye(width, dtype=np.float16)
        if style == "basis_scaled":
            values *= np.float16(65504)
    elif style in {"zeros", "maximum", "alternating", "subnormal"}:
        patterns = {"zeros": [0., -0.], "maximum": [65504.], "alternating": [65504., -65504.], "subnormal": [2**-24, -2**-24, 0., -0.]}
        values = np.resize(np.array(patterns[style], dtype=np.float16), (batch, width))
    elif style == "impulses":
        values = np.zeros((batch, width), dtype=np.float16)
        for row in range(batch):
            values[row, (row * 13 + 3) % width] = (row + 1) * (-1 if row % 2 else 1)
    else:
        bits = np.arange(65536, dtype=np.uint16)
        finite = bits[(bits & 0x7c00) != 0x7c00].view(np.float16)
        values = np.random.default_rng(plan["seed"]).choice(finite, (batch, width))
    return np.ascontiguousarray(values)


def _load(case_root, case):
    spec = importlib.util.spec_from_file_location("realfft_qualification_" + case, Path(case_root) / case / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu = cpu_audit(case_root)
    prepared = [(plan, arrays(plan)) for plan in plans()]
    cases = {}
    for case, label in EXPECTED.items():
        module = _load(case_root, case)
        records = []
        for plan, values in prepared:
            batch, width, offset = plan["batch"], plan["width"], plan["offset"]
            storage = torch.full((offset + batch * width + 31,), 42, dtype=torch.float16, device="cuda")
            source = storage[offset:offset+batch*width].view(batch, width)
            source.copy_(torch.from_numpy(values).cuda())
            before = storage.clone()
            actual = module.run(source)
            torch.cuda.synchronize()
            assert type(actual) is torch.Tensor and actual.dtype == torch.float32 and actual.device == source.device
            assert tuple(actual.shape) == (batch, width//2+1, 2) and actual.is_contiguous()
            numeric = compare(actual.cpu().numpy(), values)
            public = compare(module.reference(source).cpu().numpy(), values)
            assert public["numeric_passed"], (case, plan, public)
            preserved = torch.equal(before.view(torch.uint8), storage.view(torch.uint8))
            fresh = actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
            record = dict(plan, **numeric, legal=True, inputs_and_backing_guards_preserved=bool(preserved), output_fresh=bool(fresh), shape_dtype_valid=True)
            record["passed"] = bool(numeric["numeric_passed"] and preserved and fresh)
            records.append(record)
        failing = [row for row in records if not row["passed"]]
        cases[case] = {"ground_truth": label, "tests": records, "tests_total": len(records), "tests_passed": len(records)-len(failing),
                       "counterexamples": failing, "qualification_passed": bool(any(row["definite_failures"] for row in records) if label == "reject" else not failing and cpu["passed"]),
                       "atol": ATOL, "rtol": RTOL, "gpu_components_compared": sum(row["elements"] for row in records),
                       "comparison": "Independent direct-DFT intervals from MPFR roots, evaluated with a conservative FP64 dot-product envelope.",
                       "label_scope": "One fixed kernel under the complete finite-FP16 row domain, all three widths and all allowed batches.",
                       "oracle_dependency_hashes": dependency_hashes()}
    return {"cases": cases, "cpu_audit": cpu, "qualification_passed": all(row["qualification_passed"] for row in cases.values()), "oracle_dependency_hashes": dependency_hashes()}
