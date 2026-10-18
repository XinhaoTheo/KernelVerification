"""Exact-comparison qualification for the stable packed-key argmax pair.

The numerical oracle compares stored FP16 values directly using Python floats;
it does not use the candidates' bit ordering, packing or integer reduction.
The separate key audit verifies the finite ordering and tie invariant.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_144": "trust", "case_145": "reject"}
MAX_ROWS, MAX_COLUMNS, MAX_ELEMENTS = 65536, 4096, 262144


def finite_bit_patterns():
    bits = np.arange(65536, dtype=np.uint16)
    return bits[(bits & 0x7c00) != 0x7c00]


def independent_reference(values):
    """Exact first-maximum scan, independent of the bitwise key transform."""
    assert values.dtype == np.float16 and values.ndim == 2
    assert np.isfinite(values).all()
    result = np.zeros(values.shape[0], dtype=np.int64)
    for r, row in enumerate(values):
        best, winner = float(row[0]), 0
        for j in range(1, len(row)):
            value = float(row[j])
            if value > best:
                best, winner = value, j
        result[r] = winner
    return result


def _ranks(bits, canonicalize):
    raw = np.asarray(bits, dtype=np.uint16).astype(np.uint32)
    if canonicalize:
        raw = np.where((raw & 0x7fff) == 0, 0, raw).astype(np.uint32)
    return raw ^ np.where((raw & 0x8000) != 0, 0xffff, 0x8000).astype(np.uint32)


def key_audit():
    bits = finite_bit_patterns()
    values = bits.view(np.float16).astype(np.float64)
    order = np.argsort(values, kind="stable")
    ordered_bits, ordered_values = bits[order], values[order]
    equal = ordered_values[1:] == ordered_values[:-1]
    variants = {}
    for name, canonicalize in (("case_144", True), ("case_145", False)):
        ranks = _ranks(ordered_bits, canonicalize)
        differences = np.diff(ranks.astype(np.int64))
        strict_errors = (~equal) & (differences <= 0)
        equal_errors = equal & (differences != 0)
        failures = np.flatnonzero(strict_errors | equal_errors)
        witnesses = []
        for i in failures:
            witnesses.append({"left_bits": f"0x{int(ordered_bits[i]):04x}",
                              "right_bits": f"0x{int(ordered_bits[i+1]):04x}",
                              "left_value": float(ordered_values[i]),
                              "right_value": float(ordered_values[i+1]),
                              "left_rank": int(ranks[i]), "right_rank": int(ranks[i+1])})
        variants[name] = {"finite_encodings": len(bits),
                          "adjacent_comparisons": len(bits)-1,
                          "strict_order_failures": int(strict_errors.sum()),
                          "equal_value_rank_failures": int(equal_errors.sum()),
                          "order_and_equality_preserved": not bool(failures.size),
                          "minimum_valid_rank": int(ranks.min()),
                          "maximum_valid_rank": int(ranks.max()),
                          "witnesses": witnesses}
    indices = np.arange(MAX_COLUMNS, dtype=np.uint64)
    ties = np.uint64(0xffffffff) - indices
    assert np.all(ties[:-1] > ties[1:])
    assert np.array_equal(np.uint64(0xffffffff) - ties, indices)
    assert variants["case_144"]["order_and_equality_preserved"]
    assert variants["case_145"]["equal_value_rank_failures"] == 1
    assert all(v["minimum_valid_rank"] > 0 for v in variants.values())
    # Exhaust every legal N and the maximal M for that N. The source's
    # row*N+column and row output index remain int32-safe and in bounds.
    for n in range(1, MAX_COLUMNS+1):
        m = min(MAX_ROWS, MAX_ELEMENTS//n)
        assert (m-1)*n+(n-1) < MAX_ELEMENTS < 2**31
        block = 1 << (n-1).bit_length()
        assert n <= block <= MAX_COLUMNS
    return {"passed": True, "variants": variants,
            "tie_indices_checked": MAX_COLUMNS,
            "padding_identity": 0,
            "padding_identity_strictly_below_all_valid_keys": True,
            "packed_key_maximum": (0xfbff << 32) | 0xffffffff,
            "packed_key_fits_uint64": True,
            "all_legal_column_counts_checked": MAX_COLUMNS,
            "ordering_argument": "Checking each adjacent pair in the sorted complete finite-value set establishes strict ordering for every unequal pair by transitivity. Equal numeric values must map to equal ranks; the only distinct finite FP16 encodings with equal numeric value are signed zeros. High-32-bit rank dominates the entire low field. The descending low index field selects the earliest equal value, and integer maximum is associative. Zero padding is strictly smaller than every legal key.",
            "scope": "Exact arithmetic invariants of the audited source expressions; separate GPU checks exercise the compiled implementation, shapes and memory properties. This is not an exhaustive proof of all compiler or device executions."}


def plans():
    shapes = ((1,1),(1,2),(3,3),(4,31),(5,32),(7,33),(3,127),(2,128),
              (3,129),(5,255),(7,256),(9,257),(2,511),(3,512),(5,513),
              (2,1023),(3,1024),(5,1025),(2,2047),(3,2048),(5,2049),
              (3,4095),(64,4096),(65536,1),(65536,4))
    rows = [dict(style="random_bits",m=m,n=n,seed=144000+i,offset=(i%4)*3)
            for i,(m,n) in enumerate(shapes)]
    rows += [dict(style=style,m=63488,n=n,seed=144100,offset=7)
             for style,n in (("all_finite_singletons",1),("all_finite_equal_pairs",2),
                             ("all_finite_vs_zero",2))]
    rows += [dict(style=style,m=63487,n=2,seed=144101,offset=3)
             for style in ("adjacent_forward","adjacent_reverse")]
    rows += [dict(style="signed_zero_ties",m=4,n=n,seed=144102,offset=5)
             for n in (2,3,31,32,33,127,128,129,255,256,257,1023,1024,1025,4095,4096)]
    # Across these disjoint row batches, put the first of two equal maxima
    # at each possible column except the last, which has no later contender.
    rows += [dict(style="all_tie_positions",m=min(64,4095-start),n=4096,
                  start=start,seed=144103,offset=1)
             for start in range(0,4095,64)]
    rows.append(dict(style="last_column_max",m=64,n=4096,seed=144104,offset=13))
    return rows


def arrays(plan):
    bits = finite_bit_patterns()
    m,n = plan["m"],plan["n"]
    style = plan["style"]
    rng = np.random.default_rng(plan["seed"])
    if style == "random_bits":
        result = rng.choice(bits, size=(m,n)).view(np.float16).copy()
    elif style == "all_finite_singletons":
        result = bits.view(np.float16).reshape(-1,1).copy()
    elif style == "all_finite_equal_pairs":
        result = np.repeat(bits.view(np.float16)[:,None],2,axis=1)
    elif style == "all_finite_vs_zero":
        result = np.column_stack((bits.view(np.float16),np.zeros(len(bits),np.float16)))
    elif style.startswith("adjacent_"):
        vals = bits.view(np.float16)
        ordered = vals[np.argsort(vals.astype(np.float64),kind="stable")]
        result = np.column_stack((ordered[:-1],ordered[1:]))
        if style.endswith("reverse"):
            result = result[:,::-1].copy()
    elif style == "signed_zero_ties":
        result = np.full((m,n),-1,dtype=np.float16)
        result[0,0],result[0,-1] = -0.0,0.0
        result[1,0],result[1,-1] = 0.0,-0.0
        result[2,:],result[3,:] = -0.0,0.0
        result[2,-1],result[3,-1] = 0.0,-0.0
    elif style == "all_tie_positions":
        result = np.full((m,n),-1,dtype=np.float16)
        for r in range(m):
            first = plan["start"]+r
            result[r,first],result[r,-1] = -0.0,0.0
    elif style == "last_column_max":
        result = np.full((m,n),-65504,dtype=np.float16)
        result[:,-1] = -65504+32
    else:
        raise ValueError(style)
    assert result.shape == (m,n) and result.dtype == np.float16
    assert result.flags.c_contiguous and np.isfinite(result).all()
    return result


def _public_cpu(case_root, name):
    path = Path(case_root)/name/"kernel.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef)
                 and node.name in ("validate_inputs","reference","make_inputs")]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def cpu_audit(case_root):
    audit = key_audit()
    rows = []
    public = {name:_public_cpu(case_root,name) for name in EXPECTED}
    for plan in plans():
        values = arrays(plan)
        target = independent_reference(values)
        assert np.array_equal(target,np.argmax(values.astype(np.float64),axis=1))
        for name,ns in public.items():
            x = torch.from_numpy(values.copy())
            ns["validate_inputs"](x,require_cuda=False)
            assert np.array_equal(ns["reference"](x).numpy(),target),(name,plan)
        rows.append(dict(plan,reference_agreement=True,rows_checked=len(target)))
    witness = np.array([[-0.0,+0.0]],dtype=np.float16)
    raw = witness.view(np.uint16)
    candidate_winners = {}
    for name,canonicalize in (("case_144",True),("case_145",False)):
        ranks = _ranks(raw,canonicalize).astype(np.uint64)
        ties = np.uint64(0xffffffff)-np.arange(2,dtype=np.uint64)
        packed = (ranks << np.uint64(32)) | ties
        candidate_winners[name] = int(np.argmax(packed[0]))
    assert candidate_winners == {"case_144":0,"case_145":1}
    return {"passed":True,"key_audit":audit,"plans":rows,
            "legal_witness":{"input_bits":["0x8000","0x0000"],"reference_index":0,
                             "candidate_cpu_indices":candidate_winners},
            "oracle":"Independent exact Python first-maximum comparisons, crosschecked with NumPy numerical argmax and both public FP64 references. Key-order audit is separate from this oracle.",
            "scope":"CPU arithmetic/reference qualification; GPU execution is still required before freezing implementation labels."}


def _load(case_root,name):
    spec = importlib.util.spec_from_file_location("packed_argmax_"+name,Path(case_root)/name/"kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu = cpu_audit(case_root)
    cases = {}
    for name,label in EXPECTED.items():
        module = _load(case_root,name)
        records = []
        for plan in plans():
            values = arrays(plan)
            target = independent_reference(values)
            offset = plan["offset"]
            backing = torch.full((offset+values.size+23,),.3125,dtype=torch.float16,device="cuda")
            x = backing[offset:offset+values.size].view(values.shape)
            x.copy_(torch.from_numpy(values).cuda())
            before = backing.clone()
            legality = module.validate_inputs(x)
            out = module.run(x)
            torch.cuda.synchronize()
            valid = type(out) is torch.Tensor and out.dtype == torch.int64 and out.device == x.device and tuple(out.shape) == (values.shape[0],)
            assert valid,(name,plan,"output type/shape/device")
            actual = out.cpu().numpy()
            correct = actual == target
            preserved = torch.equal(before.view(torch.uint8),backing.view(torch.uint8))
            fresh = out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            public = module.reference(x).cpu().numpy()
            assert np.array_equal(public,target),(name,plan,"public oracle mismatch")
            mismatch = np.flatnonzero(~correct)
            witness = None
            if mismatch.size:
                r = int(mismatch[0]); expected_index = int(target[r]); actual_index = int(actual[r])
                witness = {"row":r,"expected_index":expected_index,"actual_index":actual_index,
                           "expected_value":float(values[r,expected_index]),
                           "actual_selected_value":float(values[r,actual_index]) if 0<=actual_index<values.shape[1] else None,
                           "expected_bits":f"0x{int(values.view(np.uint16)[r,expected_index]):04x}",
                           "actual_selected_bits":f"0x{int(values.view(np.uint16)[r,actual_index]):04x}" if 0<=actual_index<values.shape[1] else None}
            records.append(dict(plan,legal=bool(legality["valid"]),passed=bool(correct.all() and preserved and fresh),
                                rows_compared=len(target),mismatched_rows=int(mismatch.size),
                                max_index_error=int(np.max(np.abs(actual-target))),
                                inputs_preserved=bool(preserved),input_and_backing_guards_preserved=bool(preserved),
                                output_fresh=bool(fresh),shape_dtype_valid=True,witness=witness))
        failures = [r for r in records if not r["passed"]]
        cases[name] = {"ground_truth":label,"qualification_passed":bool(failures) if label=="reject" else not failures and cpu["passed"],
                       "tests":records,"tests_total":len(records),"tests_passed":len(records)-len(failures),
                       "counterexamples":failures,"rows_compared":sum(r["rows_compared"] for r in records),
                       "label_scope":"One exact stable-argmax verdict for the implementation across its complete declared domain.",
                       "analysis":"The correct variant preserves numeric order and numeric equality for all finite FP16 encodings; its low key field selects the first equal value and zero padding is below every legal key. GPU qualification exercises every encoding, adjacent values, every tie position, shape boundaries and offset views; this does not exhaust every compiler/device execution."}
    return {"cpu_audit":cpu,"cases":cases,"qualification_passed":all(r["qualification_passed"] for r in cases.values())}
