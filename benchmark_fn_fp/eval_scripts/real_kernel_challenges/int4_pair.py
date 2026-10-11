"""Independent canonical-format qualification for case_120 and case_121.

The numerical oracle directly decodes each public GPTQ nibble using Python
integers and evaluates FP64 outer products. It does not invoke, reconstruct or
undo the candidate's optimized repacking or grouped activation permutation.
Importing this helper runs neither a model nor a GPU service.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_120": "trust", "case_121": "reject"}
ATOL, RTOL = .01, .001


def pack_rows(codes, tail_bits=0):
    """Canonical K packing, independent scalar integer implementation."""
    k, n = codes.shape
    words = np.full(((k+7)//8, n), np.uint32(tail_bits), dtype=np.uint32)
    for row in range(k):
        shift = 4 * (row % 8)
        for col in range(n):
            old = int(words[row//8, col])
            words[row//8, col] = (old & ~(15 << shift)) | (int(codes[row,col]) << shift)
    return words.view(np.int32).copy()


def pack_zeros(zeros, tail_bits=0):
    """Canonical N packing of zero point minus one."""
    groups, n = zeros.shape
    words = np.full((groups, (n+7)//8), np.uint32(tail_bits), dtype=np.uint32)
    for group in range(groups):
        for col in range(n):
            shift = 4 * (col % 8)
            old = int(words[group,col//8])
            words[group,col//8] = (old & ~(15 << shift)) | ((int(zeros[group,col])-1) << shift)
    return words.view(np.int32).copy()


def decode_canonical(qweight, scales, qzeros, g_idx):
    """Decode logical B[k,n] without any Marlin tile or permutation knowledge."""
    k, n = len(g_idx), qweight.shape[1]
    result = np.empty((k,n), dtype=np.float16)
    for row in range(k):
        group = int(g_idx[row])
        for col in range(n):
            qword = int(qweight[row//8,col]) & 0xffffffff
            zword = int(qzeros[group,col//8]) & 0xffffffff
            q = (qword >> (4*(row%8))) & 15
            z = ((zword >> (4*(col%8))) & 15) + 1
            # FP16 scale * a small integer is exact in FP64 before the
            # explicit H16 target rounding required by the public contract.
            result[row,col] = np.float16((q-z) * float(scales[group,col]))
    return result


def independent_reference(values):
    x, qweight, scales, qzeros, g_idx, _ = values
    decoded = decode_canonical(qweight, scales, qzeros, g_idx)
    result = np.zeros((x.shape[0],qweight.shape[1]), dtype=np.float64)
    for inner in range(x.shape[1]):
        result += x[:,inner].astype(np.float64)[:,None] * decoded[inner].astype(np.float64)[None,:]
    return result


def plans():
    rows = []
    for i,(k,n) in enumerate(((1,1),(7,7),(8,8),(9,9),(15,15),(16,16),(17,17),
                              (31,31),(32,32),(33,33),(63,63),(64,64),(65,65),
                              (95,95),(96,96),(127,65),(128,96))):
        rows.append(dict(m=(1,3,17)[i%3], k=k, n=n, group_size=(16,32,64)[i%3],
                         style="random", groups="shuffled", seed=310+i))
    for start in (0,64):
        for ordering in ("ordered","reversed","shuffled"):
            rows.append(dict(m=64,k=128,n=96,group_size=16,style="basis",
                             groups=ordering,basis_start=start,seed=410,tail_bits=0))
    rows.extend([
        dict(m=65,k=128,n=96,group_size=64,style="random",groups="shuffled",seed=411),
        dict(m=17,k=65,n=65,group_size=16,style="random",groups="shuffled",seed=412,tail_bits=0),
        dict(m=17,k=65,n=65,group_size=16,style="random",groups="shuffled",seed=412,tail_bits=0xffffffff),
        dict(m=17,k=33,n=95,group_size=32,style="storage_offset",groups="reversed",seed=413),
        dict(m=3,k=128,n=96,group_size=16,style="maximum",groups="shuffled",seed=414),
        dict(m=3,k=128,n=65,group_size=16,style="subnormal_scale",groups="shuffled",seed=415),
        dict(m=3,k=128,n=65,group_size=16,style="subnormal_scale_stress",groups="shuffled",seed=419),
        dict(m=3,k=128,n=65,group_size=32,style="subnormal_x",groups="shuffled",seed=416),
        dict(m=1,k=17,n=65,group_size=16,style="zero",groups="shuffled",seed=417),
        dict(m=17,k=127,n=96,group_size=16,style="varying_parameters",groups="reversed",seed=418),
    ])
    return rows


def arrays(plan):
    rng = np.random.default_rng(plan["seed"])
    m,k,n,gs = (plan[key] for key in ("m","k","n","group_size"))
    groups = (k+gs-1)//gs
    g_idx = (np.arange(k)//gs).astype(np.int32)
    if plan["groups"] == "reversed": g_idx = g_idx[::-1].copy()
    elif plan["groups"] == "shuffled": g_idx = rng.permutation(g_idx)
    x = rng.uniform(-.5,.5,(m,k)).astype(np.float16)
    codes = rng.integers(0,16,(k,n),dtype=np.int32)
    zeros = rng.integers(1,17,(groups,n),dtype=np.int32)
    scales = rng.uniform(2**-10,2**-5,(groups,n)).astype(np.float16)
    if plan["style"] == "basis":
        x.fill(0)
        for i in range(m): x[i,plan["basis_start"]+i] = .5
        codes = ((np.arange(k)[:,None]+11*np.arange(n)[None,:])%16).astype(np.int32)
        zeros.fill(8)
        scales.fill(1/32)
    elif plan["style"] == "maximum":
        x.fill(.5)
        codes.fill(0)
        zeros.fill(16)
        scales.fill(1/32)
    elif plan["style"] == "subnormal_scale":
        x.fill(.5)
        scales[:] = np.where(np.arange(n)[None,:]%2, 2**-24, 2**-14-2**-24)
    elif plan["style"] == "subnormal_x":
        x.fill(2**-14-2**-24)
        codes.fill(0)
        zeros.fill(16)
        scales.fill(1/32)
    elif plan["style"] == "subnormal_scale_stress":
        x.fill(.5)
        x[1].fill(-.5)
        codes.fill(0)
        zeros.fill(16)
        scales.fill(2**-14-2**-24)
    elif plan["style"] == "zero":
        x.fill(0)
    elif plan["style"] == "varying_parameters":
        codes = ((3*np.arange(k)[:,None]+np.arange(n)[None,:])%16).astype(np.int32)
        zeros = ((np.arange(groups)[:,None]+5*np.arange(n)[None,:])%16+1).astype(np.int32)
        scales = np.power(2.,-5-(np.arange(groups)[:,None]+np.arange(n)[None,:])%4).astype(np.float16)
    tail = plan.get("tail_bits",0x98765432)
    return x,pack_rows(codes,tail),scales,pack_zeros(zeros,tail),g_idx,gs


def layout_audit():
    """Finite ABI bijection check, separate from the numerical oracle.

    Compare the source layout's explicit permutation enumeration against the
    consumer's independently derived coordinate formula over the full tile.
    This diagnostic never participates in decode_canonical or the reference.
    """
    source = []
    for i in range(32):
        lane_rows = (2*(i%4), 2*(i%4)+1, 2*(i%4+4), 2*(i%4+4)+1)
        local = [16*r+i//4+8*block for block in (0,1) for r in lane_rows]
        for j in range(4): source.extend(v+256*j for v in local)
    order = [0,2,4,6,1,3,5,7]
    inverse = [order.index(i) for i in range(8)]
    correct, mutated = [], []
    for begin in range(0,len(source),8):
        correct.extend(source[begin+idx] for idx in order)
        mutated.extend(source[begin+idx] for idx in inverse)
    assert sorted(correct) == list(range(1024))
    slots = []
    mismatches = 0
    for k in range(16):
        for n in range(64):
            word = 4*(4*(n%8)+(k%8)//2)+(n%64)//16
            lane = 4*(k%2)+2*((n%16)//8)+(k%16)//8
            slot = word*8+lane
            canonical = (n//16)*256+k*16+n%16
            assert correct[slot] == canonical, (k,n,slot)
            mismatches += mutated[slot] != canonical
            slots.append(slot)
    assert sorted(slots) == list(range(1024)) and mismatches > 0
    # Verify canonical code packing for every code, both packing axes, all
    # eight nibble lanes and signed int32 high-bit patterns.
    count = 0
    for code in range(16):
        codes = np.full((8,8),code,np.int32)
        qw = pack_rows(codes,0xffffffff)
        zero = np.full((1,8),16-code,np.int32)
        qz = pack_zeros(zero,0xffffffff)
        decoded = decode_canonical(qw,np.full((1,8),1/32,np.float16),qz,np.zeros(8,np.int32))
        assert np.array_equal(decoded,np.full((8,8),(2*code-16)/32,np.float16))
        if code >= 8: assert (qw<0).all()
        if 15-code >= 8: assert (qz<0).all()
        count += 64
    return {"passed":True,"tile_coordinates_checked":1024,
            "consumer_slots_bijective":True,"inverse_interleave_mismatches":int(mismatches),
            "canonical_code_lane_checks":count,"all_16_codes":True,
            "signed_high_bit_words":True,
            "scope":"integer layout and independent canonical encode/decode checks; does not prove floating point or all runtime behavior"}


def _public_cpu(case_root, name):
    path = case_root/name/"kernel.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node,ast.FunctionDef)
                and node.name in {"validate_inputs","reference","make_inputs"}]
    namespace = {"torch":torch}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def _host(values):
    return tuple(torch.from_numpy(value.copy()) for value in values[:-1])+(values[-1],)


def cpu_audit(case_root:Path):
    public = _public_cpu(case_root,"case_120")
    rows = []
    for plan in plans():
        values = arrays(plan)
        host = _host(values)
        public["validate_inputs"](*host,require_cuda=False)
        target = independent_reference(values)
        direct = np.einsum("ik,kn->in",values[0].astype(np.float64),
                           decode_canonical(*values[1:5]).astype(np.float64),optimize=False)
        public_ref = public["reference"](*host).numpy()
        assert all(np.isfinite(v).all() for v in (target,direct,public_ref))
        error = max(float(np.max(np.abs(target-direct))),float(np.max(np.abs(target-public_ref))))
        assert error < 1e-11,(plan,error)
        rows.append(dict(plan,reference_agreement_max_abs=error))
    base = dict(m=17,k=65,n=65,group_size=16,style="random",groups="shuffled",seed=412)
    zero_tail = independent_reference(arrays(dict(base,tail_bits=0)))
    high_tail = independent_reference(arrays(dict(base,tail_bits=0xffffffff)))
    assert np.array_equal(zero_tail,high_tail)
    invalid=[]
    for kind in ("x_bound","x_nan","scale_zero","scale_bound","scale_nan","gidx_range",
                 "gidx_counts","group_size","bool_group_size","qweight_dtype","grad",
                 "negative_view","inference_tensor","overlap","noncontiguous"):
        host=list(_host(arrays(dict(m=3,k=17,n=17,group_size=16,style="random",groups="ordered",seed=513))))
        if kind=="x_bound": host[0][0,0]=1
        elif kind=="x_nan": host[0][0,0]=float("nan")
        elif kind=="scale_zero": host[2][0,0]=0
        elif kind=="scale_bound": host[2][0,0]=.5
        elif kind=="scale_nan": host[2][0,0]=float("nan")
        elif kind=="gidx_range": host[4][0]=2
        elif kind=="gidx_counts": host[4][0]=1
        elif kind=="group_size": host[5]=17
        elif kind=="bool_group_size": host[5]=True
        elif kind=="qweight_dtype": host[1]=host[1].long()
        elif kind=="grad": host[0].requires_grad_(True)
        elif kind=="negative_view": host[0]=torch._neg_view(host[0])
        elif kind=="inference_tensor":
            with torch.inference_mode(): host[0]=host[0].clone()
        elif kind=="overlap": host[3]=host[1].flatten()[:host[3].numel()].reshape(host[3].shape)
        elif kind=="noncontiguous": host[0]=host[0].T.contiguous().T
        try: public["validate_inputs"](*host,require_cuda=False)
        except ValueError as exc: invalid.append({"kind":kind,"rejected":True,"reason":str(exc)})
        else: raise AssertionError(f"Public input validator accepted {kind}")
    return {"passed":True,"plans":rows,"layout":layout_audit(),
            "invalid_inputs":invalid,
            "canonical_tail_bits_invariance":True,
            "method":"Python uint32 nibble decode and explicit H16 dequantization, serial FP64 outer products, independent NumPy einsum and public FP64 reference; no candidate repacking or grouped gather is used in the oracle."}


def _load(case_root,name):
    modname="int4_qualification_"+name
    spec=importlib.util.spec_from_file_location(modname,case_root/name/"kernel.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[modname]=module
    spec.loader.exec_module(module)
    return module


def validate(case_root:Path):
    cpu=cpu_audit(case_root)
    cases={}
    for name,expected in EXPECTED.items():
        module=_load(case_root,name)
        records=[]
        tail_outputs=[]
        for plan in plans():
            values=arrays(plan)
            target=independent_reference(values)
            tensors=[]
            for value in values[:-1]:
                tensor=torch.from_numpy(value.copy()).cuda()
                if plan["style"]=="storage_offset":
                    base=torch.empty(tensor.numel()+17,dtype=tensor.dtype,device="cuda")
                    view=base[5:5+tensor.numel()].reshape(tensor.shape)
                    view.copy_(tensor)
                    tensor=view
                tensors.append(tensor)
            gpu=tuple(tensors)+(values[-1],)
            legality=module.validate_inputs(*gpu)
            before=tuple(t.clone() for t in tensors)
            actual=module.run(*gpu)
            torch.cuda.synchronize()
            public_ref=module.reference(*gpu).cpu().numpy()
            assert np.max(np.abs(public_ref-target)) < 1e-11
            array=actual.double().cpu().numpy()
            valid_type=(actual.dtype==torch.float32 and actual.device==tensors[0].device
                        and tuple(actual.shape)==target.shape)
            finite=bool(np.isfinite(array).all())
            preserved=all(torch.equal(old.view(torch.uint8),new.view(torch.uint8))
                          for old,new in zip(before,tensors))
            left,right=actual.data_ptr(),actual.data_ptr()+actual.numel()*actual.element_size()
            fresh=all(right<=t.data_ptr() or t.data_ptr()+t.numel()*t.element_size()<=left for t in tensors)
            delta=np.abs(array-target)
            ratios=delta/(ATOL+RTOL*np.abs(target))
            ratio=float(np.max(ratios))
            witness=None
            if ratio>1:
                index=np.unravel_index(np.argmax(ratios),ratios.shape)
                witness={"index":list(map(int,index)),"actual":float(array[index]),
                         "reference":float(target[index]),"absolute_error":float(delta[index]),
                         "allowed_error":float(ATOL+RTOL*abs(target[index])),"ratio":float(ratios[index])}
            records.append(dict(plan,legal=bool(legality.get("valid",True)),passed=bool(valid_type and finite and preserved and fresh and ratio<=1),
                max_error_ratio=ratio,max_abs_error=float(np.max(delta)),finite=finite,
                inputs_preserved=preserved,shape_dtype_valid=bool(valid_type),output_fresh=bool(fresh),witness=witness))
            if plan.get("seed")==412:tail_outputs.append(array.copy())
        tails_equal=len(tail_outputs)==2 and np.array_equal(*tail_outputs)
        failures=[r for r in records if not r["passed"]]
        cases[name]={"ground_truth":expected,"qualification_passed":bool(failures) if expected=="reject" else not failures and tails_equal,
            "tests":records,"tests_total":len(records),"tests_passed":len(records)-len(failures),
            "counterexamples":failures,"canonical_tail_bits_invariance":bool(tails_equal),
            "label_scope":"fixed implementation under the entire public canonical-format contract",
            "analysis":"Independent scalar canonical decode plus explicit H16 and FP64 contraction; basis inputs cover all K rows and N columns of a128x96 decoded matrix. Integer layout enumeration is separate from the numerical oracle; finite qualification is not a formal full-program proof."}
    return {"cases":cases,"cpu_audit":cpu,"qualification_passed":all(c["qualification_passed"] for c in cases.values())}
