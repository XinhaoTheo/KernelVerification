"""Independent integer qualification for Winograd tile cases160/161.

The oracle is direct Python-integer cross-correlation. CPU checks execute
an AST adaptation of the actual candidate and cover all36x9 bilinear basis
pairs. GPU checks independently compare the compiled source with the oracle.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_87": "trust", "case_88": "reject"}


def dependency_hashes():
    path = Path(__file__)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


def independent_reference(tiles, filters):
    tiles, filters = np.asarray(tiles), np.asarray(filters)
    batch = len(tiles)
    assert tiles.shape == (batch, 6, 6) and filters.shape == (batch, 3, 3)
    assert tiles.dtype == filters.dtype == np.int8
    out = np.empty((batch, 4, 4), dtype=np.int32)
    for b in range(batch):
        for r in range(4):
            for c in range(4):
                out[b, r, c] = sum(int(tiles[b, r+i, c+j]) * int(filters[b, i, j])
                                   for i in range(3) for j in range(3))
    return out


def basis_arrays():
    tiles = np.zeros((324, 6, 6), dtype=np.int8)
    filters = np.zeros((324, 3, 3), dtype=np.int8)
    for x in range(36):
        for w in range(9):
            tiles[x*9+w].flat[x] = 1
            filters[x*9+w].flat[w] = 1
    return tiles, filters


class _Pointer:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def __add__(self, offset):
        return _Pointer(self.data, self.offset + offset)


class _TL:
    """CPU integer adapter; it is not evidence about GPU compilation."""
    int32, int64 = np.int32, np.int64

    @staticmethod
    def arange(start, stop):
        return np.arange(start, stop, dtype=np.int32)

    @staticmethod
    def program_id(axis):
        assert axis == 0
        return 0

    @staticmethod
    def load(pointer, mask, other):
        result = np.full(len(mask), other, dtype=pointer.data.dtype)
        offsets = np.asarray(pointer.offset)
        assert np.all((offsets[mask] >= 0) & (offsets[mask] < pointer.data.size))
        result[mask] = pointer.data[offsets[mask]]
        return result

    @staticmethod
    def store(pointer, value, mask):
        offsets = np.asarray(pointer.offset)
        assert np.all((offsets[mask] >= 0) & (offsets[mask] < pointer.data.size))
        pointer.data[offsets[mask]] = value[mask]


class _Casts(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.FloorDiv):
            return ast.copy_location(ast.Call(func=ast.Name(id="_trunc_div", ctx=ast.Load()), args=[node.left, node.right], keywords=[]), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "to" and node.args
                and isinstance(node.args[0], ast.Attribute) and isinstance(node.args[0].value, ast.Name)
                and node.args[0].value.id == "tl"):
            return ast.copy_location(ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="asarray", ctx=ast.Load()),
                                              args=[node.func.value], keywords=[ast.keyword(arg="dtype", value=node.args[0])]), node)
        return node


def _source_cpu(case_root, name):
    path = Path(case_root) / name / "kernel.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != "run"]
    for fn in functions:
        fn.decorator_list = []
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        if fn.name == "_tiles":
            fn.body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id=f"y{r}{c}", ctx=ast.Load())
                                                            for r in range(4) for c in range(4)], ctx=ast.Load())))
    module = ast.fix_missing_locations(_Casts().visit(ast.Module(body=functions, type_ignores=[])))
    namespace = {"np": np, "torch": torch, "tl": _TL,
                 "_trunc_div": lambda a, b: np.where(a >= 0, a // b, -((-a) // b))}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _emulate(namespace, tiles, filters):
    batch = len(tiles)
    block = 1 << (batch - 1).bit_length()
    output = np.full(batch*16, -99999, dtype=np.int32)
    numerator_columns = namespace["_tiles"](_Pointer(tiles.ravel()), _Pointer(filters.ravel()),
                                            _Pointer(output), batch, block)
    numerator = np.stack(numerator_columns, axis=1)[:batch].reshape(batch, 4, 4)
    return output.reshape(batch, 4, 4), numerator


def arithmetic_audit():
    # Actual candidate helper coefficients, after moving G denominators to A.
    data_l1, weight_l1, output_l1 = 10, 7, 48
    data_bound = 128 * data_l1**2
    weight_bound = 128 * weight_l1**2
    product_bound = data_bound * weight_bound
    inverse_bound = product_bound * output_l1**2
    direct_bound = 9 * 128**2
    assert data_bound < 2**31 and weight_bound < 2**31
    assert product_bound < 2**31 and inverse_bound < 2**63
    assert direct_bound < 2**31
    return {"passed": True, "data_transform_abs_bound": data_bound,
            "filter_transform_abs_bound": weight_bound, "product_abs_bound": product_bound,
            "inverse_intermediate_abs_bound": inverse_bound, "output_abs_bound": direct_bound,
            "proof": [
                "Both input transforms are integer linear maps. The largest row absolute coefficient sum is10 for data and7 for filters. Two spatial axes give bounds128*10^2 and128*7^2. All INT32 transform intermediates are safe.",
                "The kernel casts both transformed operands to INT64 before their elementwise product. The product is bounded by80281600. Each inverse-transform row has absolute coefficient sum at most48; both axes give184968806400<2^63. These bounds also cover every partial sum.",
                "Before final division, the actual-source arithmetic is bilinear in36 tile values and9 filter values. Exact checking of all324 input/filter basis pairs proves its coefficient tensor. For the control, every basis numerator is576 times direct cross-correlation. Therefore the identity and divisibility hold for every integer input pair, including the whole INT8 domain.",
                "The exact final result is a sum of9 products and has absolute value at most147456. It fits INT32. Source binding, bounds and complete bilinear basis checks support the control label. GPU tests check compilation and tensor behavior separately."
            ]}


def _source_binding(case_root, name, namespace):
    path = Path(case_root) / name / "kernel.py"
    data = np.array([namespace["_data_transform"](*np.eye(6, dtype=np.int64)[i]) for i in range(6)]).T
    weight = np.array([namespace["_filter_transform"](*np.eye(3, dtype=np.int64)[i]) for i in range(3)]).T
    inverse = np.array([namespace["_output_transform"](*np.eye(6, dtype=np.int64)[i]) for i in range(6)]).T
    assert max(abs(data).sum(1)) == 10
    assert max(abs(weight).sum(1)) == 7
    assert max(abs(inverse).sum(1)) == 48
    tree = ast.parse(path.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tiles")
    products = [n for n in kernel.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("m")]
    assert len(products) == 36
    for node in products:
        expression = ast.unparse(node.value)
        assert expression.count(".to(tl.int64)") == 2 and " * " in expression
    # Verify the three helpers are linear expressions with no conditional or nonlinear operation.
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_data_transform", "_filter_transform", "_output_transform"}]:
        for node in ast.walk(fn):
            assert not isinstance(node, (ast.If, ast.For, ast.While, ast.IfExp, ast.Call, ast.Pow, ast.Div, ast.FloorDiv))
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                def literal(value):
                    return isinstance(value, ast.Constant) or (isinstance(value, ast.UnaryOp) and isinstance(value.operand, ast.Constant))
                assert literal(node.left) or literal(node.right)
    return {"kernel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_matrix": data.tolist(), "filter_matrix": weight.tolist(), "scaled_output_matrix": inverse.tolist(),
            "source_helpers_are_integer_linear": True, "products_cast_to_int64": True,
            "indexing": "Lane j of program p reads tile16*p+j. Loads use36*tile+[0,35] and9*tile+[0,8]. Stores use16*tile+[0,15]. All use tile<B masks. B<=2048 makes every offset INT32-safe. There are no cross-lane dependencies. Tensor pointers include any storage offset."}


def plans():
    rows = [dict(batch=n, style="random", seed=160000+i, offset_x=i % 11, offset_w=(i+3) % 13)
            for i, n in enumerate([1, 2, 3, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025, 2047, 2048])]
    rows += [dict(batch=324, style="basis", seed=160100, offset_x=3, offset_w=5),
             dict(batch=1, style="witness", seed=160101, offset_x=7, offset_w=11),
             dict(batch=81, style="extremes", seed=160102, offset_x=13, offset_w=17),
             dict(batch=65, style="alias", seed=160103, offset_x=19, offset_w=24)]
    return rows


def arrays(plan):
    batch, style = plan["batch"], plan["style"]
    rng = np.random.default_rng(plan["seed"])
    tiles = rng.integers(-128, 128, size=(batch, 6, 6), dtype=np.int16).astype(np.int8)
    filters = rng.integers(-128, 128, size=(batch, 3, 3), dtype=np.int16).astype(np.int8)
    if style == "basis":
        return basis_arrays()
    if style == "witness":
        tiles.fill(0); filters.fill(0)
        tiles[0, 1, 1] = 72
        filters[0, 1, 1] = 72
    elif style == "extremes":
        values = np.array([-128, -127, -1, 0, 1, 126, 127], dtype=np.int8)
        tiles = rng.choice(values, size=tiles.shape)
        filters = rng.choice(values, size=filters.shape)
        tiles[0].fill(-128); filters[0].fill(-128)
        tiles[1] = ((np.indices((6, 6)).sum(0) % 2) * 255 - 128).astype(np.int8)
        filters[1] = ((np.indices((3, 3)).sum(0) % 2) * 255 - 128).astype(np.int8)
        tiles[2].fill(127); filters[2].fill(127)
    elif style == "alias":
        # Two contiguous views overlap in one backing storage. Their shapes differ.
        offset = plan["offset_w"] - plan["offset_x"]
        filters = tiles.ravel()[offset:offset+batch*9].reshape(batch, 3, 3).copy()
    return tiles, filters


def cpu_audit(case_root):
    arithmetic = arithmetic_audit()
    tiles, filters = basis_arrays()
    reference = independent_reference(tiles, filters)
    cases = {}
    for name, expected in EXPECTED.items():
        namespace = _source_cpu(case_root, name)
        binding = _source_binding(case_root, name, namespace)
        actual, numerator = _emulate(namespace, tiles, filters)
        exact_identity = numerator == reference.astype(np.int64) * 576
        public = namespace["reference"](torch.from_numpy(tiles), torch.from_numpy(filters)).numpy()
        assert np.array_equal(public, reference)
        if expected == "trust":
            assert bool(exact_identity.all()) and np.array_equal(actual, reference)
        else:
            assert not bool(exact_identity.all()) and not np.array_equal(actual, reference)
        sample_checks = []
        for plan in plans():
            x, w = arrays(plan)
            target = independent_reference(x, w)
            observed, num = _emulate(namespace, x, w)
            public = namespace["reference"](torch.from_numpy(x), torch.from_numpy(w)).numpy()
            assert np.array_equal(public, target)
            if expected == "trust":
                assert np.array_equal(observed, target) and np.all(num % 576 == 0)
            sample_checks.append({"style": plan["style"], "batch": plan["batch"], "different_outputs": int(np.count_nonzero(observed != target))})
        witness_plan = next(p for p in plans() if p["style"] == "witness")
        x, w = arrays(witness_plan)
        observed, num = _emulate(namespace, x, w)
        target = independent_reference(x, w)
        cases[name] = {"source_binding": binding, "bilinear_basis_pairs": 324,
                       "basis_output_coefficients": 5184, "basis_numerator_identity_failures": int(np.count_nonzero(~exact_identity)),
                       "basis_output_mismatches": int(np.count_nonzero(actual != reference)),
                       "public_reference_matches_python_integer_oracle": True,
                       "sample_checks": sample_checks,
                       "cpu_witness": {"tiles": x.tolist(), "filters": w.tolist(), "actual_cpu_model": observed.tolist(),
                                       "reference": target.tolist(), "all_numerators_divisible_by576": bool(np.all(num % 576 == 0)),
                                       "actual_gpu_evidence": False}}
    return {"passed": True, "cases": cases, "arithmetic": arithmetic,
            "oracle": "Direct nine-term cross-correlation with Python arbitrary-precision integers. It uses none of the candidate's transforms.",
            "cpu_execution_limit": "AST adapter executes the actual kernel arithmetic and pointer expressions with NumPy integers. Signed division is adapted to truncation toward zero. The Accept basis proves exact divisibility, and the recorded reject witness is also divisible. Only actual GPU results qualify compiler behavior.",
            "oracle_dependency_hashes": dependency_hashes()}


def _load(case_root, name):
    spec = importlib.util.spec_from_file_location("winograd_qualification_" + name, Path(case_root)/name/"kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu = cpu_audit(case_root)
    prepared = [(plan, *arrays(plan)) for plan in plans()]
    oracle = [independent_reference(x, w) for _, x, w in prepared]
    cases = {}
    for name, expected in EXPECTED.items():
        module = _load(case_root, name)
        records = []
        for (plan, x_np, w_np), target in zip(prepared, oracle):
            batch, ox, ow = plan["batch"], plan["offset_x"], plan["offset_w"]
            bx = torch.full((ox + batch*36 + 31,), 42, dtype=torch.int8, device="cuda")
            bw = torch.full((ow + batch*9 + 37,), -42, dtype=torch.int8, device="cuda")
            x = bx[ox:ox+batch*36].view(batch, 6, 6)
            w = bw[ow:ow+batch*9].view(batch, 3, 3)
            x.copy_(torch.from_numpy(x_np).cuda())
            w.copy_(torch.from_numpy(w_np).cuda())
            if plan["style"] == "alias":
                bw = bx
                w = bx[ow:ow+batch*9].view(batch, 3, 3)
                assert torch.equal(w.cpu(), torch.from_numpy(w_np))
            before_x, before_w = bx.clone(), bw.clone()
            actual = module.run(x, w)
            torch.cuda.synchronize()
            shape_valid = type(actual) is torch.Tensor and actual.dtype == torch.int32 and actual.device == x.device and tuple(actual.shape) == (batch, 4, 4)
            assert shape_valid
            observed = actual.cpu().numpy()
            public = module.reference(x, w).cpu().numpy()
            assert np.array_equal(public, target), (name, plan, "public_reference")
            failed = observed != target
            count = int(np.count_nonzero(failed))
            preserved = torch.equal(bx, before_x) and torch.equal(bw, before_w)
            fresh = all(actual.untyped_storage().data_ptr() != t.untyped_storage().data_ptr() for t in (x, w))
            witness = None
            if count:
                b, r, c = map(int, np.argwhere(failed)[0])
                witness = {"index": [b, r, c], "tile": x_np[b].tolist(), "filter": w_np[b].tolist(),
                           "actual": int(observed[b, r, c]), "reference": int(target[b, r, c])}
            records.append(dict(plan, passed=bool(not count and preserved and fresh), numeric_passed=count == 0,
                                legal=True, definite_failures=count, elements=batch*16,
                                max_abs_error=int(np.abs(observed.astype(np.int64)-target).max()),
                                inputs_and_backing_guards_preserved=bool(preserved), output_fresh=bool(fresh),
                                shape_dtype_valid=bool(shape_valid), witness=witness))
        failed_plans = [p for p in records if not p["passed"]]
        cases[name] = {"ground_truth": expected, "tests": records, "tests_total": len(records),
                       "tests_passed": len(records)-len(failed_plans), "counterexamples": failed_plans,
                       "qualification_passed": bool(any(p["definite_failures"] for p in records) if expected == "reject" else not failed_plans and cpu["passed"]),
                       "atol": 0, "rtol": 0, "comparison": "exact integer equality against direct Python-integer cross-correlation",
                       "gpu_outputs_compared": sum(p["elements"] for p in records),
                       "oracle_dependency_hashes": dependency_hashes(),
                       "label_scope": "One fixed implementation over all legal INT8 tiles and filters in the complete public contract.",
                       "analysis": "Complete bilinear basis and source-bound no-overflow arguments support the control. Actual GPU tests cover that complete basis, extremes, random values, maximum batch, block tails, storage offsets, overlapping inputs, guard preservation and fresh output storage."}
    return {"cases": cases, "cpu_audit": cpu, "oracle_dependency_hashes": dependency_hashes(),
            "qualification_passed": all(c["qualification_passed"] for c in cases.values())}
