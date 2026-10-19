"""Independent finite-domain diffusion oracle for case_49 and case_50.

The numerical reference explicitly applies the public nine-neighbour update
at every time step. It never uses the candidate's fused coefficients or image
extension. The separate integer matrix audit checks the proposed fusion law.
Importing this module makes no model, network or GPU calls.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_49": "trust", "case_50": "reject"}
ATOL = 2e-5


def independent_reference(x, steps):
    current = np.asarray(x, dtype=np.float64).copy()
    h, w = current.shape
    for _ in range(steps):
        padded = np.pad(current, ((1, 1), (1, 1)), mode="constant")
        result = np.zeros_like(current)
        for dr, wr in enumerate((1, 2, 1)):
            for dc, wc in enumerate((1, 2, 1)):
                result += (wr * wc / 16.) * padded[dr:dr+h, dc:dc+w]
        current = result
    return current


def integer_matrix_audit():
    """Every legal axis length, every input/output coordinate, exact integers.

    The one-dimensional unscaled operator has diagonal 2 and neighbours 1.
    Build its fourth power by four nearest-neighbour updates, independently
    of the fused implementation. Compare with a signed image expansion.
    """
    coeff = (1, 8, 28, 56, 70, 56, 28, 8, 1)
    entries = 0
    incorrect_lengths = []
    for n in range(1, 130):
        power = np.eye(n, dtype=np.int64)
        for _ in range(4):
            next_power = 2 * power
            next_power[1:] += power[:-1]
            next_power[:-1] += power[1:]
            power = next_power
        images = np.zeros((n, n), dtype=np.int64)
        zero_once = np.zeros((n, n), dtype=np.int64)
        for row in range(n):
            for offset, weight in zip(range(-4, 5), coeff):
                q = (row + offset + 1) % (2 * (n + 1))
                if 1 <= q <= n:
                    images[row, q-1] += weight
                elif n + 2 <= q <= 2*n + 1:
                    images[row, 2*n+1-q] -= weight
                if 0 <= row + offset < n:
                    zero_once[row, row+offset] += weight
        assert np.array_equal(images, power), n
        assert np.all(power >= 0) and np.all(power.sum(axis=1) <= 256)
        if not np.array_equal(zero_once, power):
            incorrect_lengths.append(n)
        entries += n*n
    assert incorrect_lengths == list(range(1, 130))
    u = 2.**-24
    gamma18 = (18*u)/(1-18*u)
    bound = (1+gamma18)**4-1
    assert bound < ATOL
    return {"passed": True, "axis_lengths": 129,
            "exact_matrix_entries_checked": entries,
            "zero_once_mismatching_lengths": incorrect_lengths,
            "rounding_absolute_error_bound": bound,
            "rounding_derivation": "Each axis pass has 9 weighted signed loads, absolute coefficient sum 1; <=18 FP32 operations gives gamma18. At most 4 passes for 8 steps gives (1+gamma18)^4-1 for |x|<=1. FP16 inputs convert exactly; dyadic coefficients and <=4 passes stay far above FP32 underflow. This bounds arithmetic only, not memory/runtime correctness.",
            "scope": "Full discrete axis-size domain for the modeled linear operators. Two-dimensional equivalence follows from commuting row/column operators; four/eight-step composition checked separately from runtime."}


def plans():
    shapes = ((1,1),(1,2),(2,1),(2,2),(3,5),(7,9),(8,8),(9,7),
              (15,17),(16,16),(17,15),(31,33),(32,32),(33,31),
              (63,65),(64,64),(65,63),(127,129),(129,127),(129,129))
    rows = [dict(h=h,w=w,steps=s,style="random",seed=600+i)
            for i,(h,w) in enumerate(shapes) for s in (4,8)]
    for style in ("ones","negative_ones","zeros","checkerboard","corner",
                  "center","edge","subnormal","signed_zero","normal_min"):
        for s in (4,8):
            rows.append(dict(h=17,w=19,steps=s,style=style,seed=700))
    for s in (4,8):
        rows.append(dict(h=33,w=65,steps=s,style="storage_offset",seed=701))
    return rows


def arrays(plan):
    h,w = plan["h"],plan["w"]
    x = np.random.default_rng(plan["seed"]).uniform(-1,1,(h,w)).astype(np.float16)
    style = plan["style"]
    if style in {"ones","negative_ones","zeros","subnormal","normal_min"}:
        x.fill({"ones":1,"negative_ones":-1,"zeros":0,
                "subnormal":2**-24,"normal_min":2**-14}[style])
    elif style == "checkerboard":
        x[:] = 2*((np.arange(h)[:,None]+np.arange(w)[None,:])%2)-1
    elif style in {"corner","center","edge"}:
        x.fill(0)
        index = {"corner":(0,0),"center":(h//2,w//2),"edge":(0,w//2)}[style]
        x[index] = 1
    elif style == "signed_zero":
        x.fill(0)
        x[:,::2] = -0.
    return x


def _public_cpu(case_root, name):
    path = case_root/name/"kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"validate_inputs", "reference", "make_inputs"}]
    namespace = {"torch":torch}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def cpu_audit(case_root):
    matrix = integer_matrix_audit()
    refs = []
    for name in EXPECTED:
        public = _public_cpu(case_root,name)
        largest = 0.
        for plan in plans():
            x = arrays(plan)
            host = torch.from_numpy(x.copy())
            public["validate_inputs"](host,plan["steps"],require_cuda=False)
            actual = public["reference"](host,plan["steps"]).numpy()
            target = independent_reference(x,plan["steps"])
            error = float(np.max(np.abs(actual-target)))
            largest = max(largest,error)
            assert error < 1e-12,(name,plan,error)
        refs.append({"case":name,"plans":len(plans()),"max_reference_error":largest})
    return {"passed":True,"matrix":matrix,"public_references":refs,
            "method":"Independent repeated 3x3 FP64 update with zero exterior reapplied at every step; separate integer operator identity covers every legal axis length."}


def _load(case_root,name):
    modname = "stencil_qualification_"+name
    spec = importlib.util.spec_from_file_location(modname,case_root/name/"kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root:Path):
    cpu = cpu_audit(case_root)
    cases = {}
    for name,expected in EXPECTED.items():
        module = _load(case_root,name)
        records = []
        for plan in plans():
            values = arrays(plan)
            target = independent_reference(values,plan["steps"])
            x = torch.from_numpy(values.copy()).cuda()
            if plan["style"] == "storage_offset":
                base = torch.empty(x.numel()+23,device="cuda",dtype=x.dtype)
                view = base[7:7+x.numel()].reshape(x.shape)
                view.copy_(x)
                x = view
            module.validate_inputs(x,plan["steps"])
            before = x.clone()
            actual = module.run(x,plan["steps"])
            torch.cuda.synchronize()
            array = actual.double().cpu().numpy()
            public_ref = module.reference(x,plan["steps"]).cpu().numpy()
            assert np.max(np.abs(public_ref-target)) < 1e-12
            valid_type = actual.dtype==torch.float32 and actual.device==x.device and tuple(actual.shape)==target.shape
            finite = bool(np.isfinite(array).all())
            preserved = torch.equal(before.view(torch.uint8),x.view(torch.uint8))
            left,right = actual.data_ptr(),actual.data_ptr()+actual.numel()*actual.element_size()
            fresh = right<=x.data_ptr() or x.data_ptr()+x.numel()*x.element_size()<=left
            delta = np.abs(array-target)
            error = float(np.max(delta))
            witness = None
            if error > ATOL:
                index = np.unravel_index(np.argmax(delta),delta.shape)
                witness = {"index":list(map(int,index)),"actual":float(array[index]),
                           "reference":float(target[index]),"absolute_error":float(delta[index]),
                           "allowed_error":ATOL,"ratio":float(delta[index]/ATOL)}
            records.append(dict(plan,legal=True,passed=bool(valid_type and finite and preserved and fresh and error<=ATOL),
                max_error_ratio=error/ATOL,max_abs_error=error,finite=finite,
                inputs_preserved=bool(preserved),shape_dtype_valid=bool(valid_type),output_fresh=bool(fresh),witness=witness))
        failures = [row for row in records if not row["passed"]]
        cases[name] = {"ground_truth":expected,
            "qualification_passed":bool(failures) if expected=="reject" else not failures and cpu["passed"],
            "tests":records,"tests_total":len(records),"tests_passed":len(records)-len(failures),
            "counterexamples":failures,
            "label_scope":"fixed implementation over the entire public zero-boundary diffusion contract",
            "analysis":"Exact finite-axis operator identity, conservative FP32 error bound and independent GPU numerical/contract checks. A passing finite GPU suite alone is not a full-program proof."}
    return {"cases":cases,"cpu_audit":cpu,
            "qualification_passed":all(row["qualification_passed"] for row in cases.values())}
