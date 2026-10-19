"""Private, independent qualification for routed grouped-GEMM case_45/119.

Importing this module starts no service. cpu_audit() is offline; validate()
uses the caller's CUDA device. Its plans are fixed before model evaluation.
"""
from __future__ import annotations

import ast
from collections import Counter
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

EXPECTED = {"case_45": "trust", "case_46": "reject"}


def plans():
    return [
        dict(m=1,e=1,q=1,k=1,n=1,route="single",style="marker",seed=1),
        dict(m=1,e=1,q=1,k=1,n=33,route="single",style="marker",seed=2),
        dict(m=16,e=4,q=1,k=31,n=32,route="balanced",style="random",seed=3),
        dict(m=17,e=5,q=1,k=33,n=33,route="balanced",style="marker",seed=4),
        dict(m=65,e=1,q=1,k=1,n=33,route="single",style="marker",seed=5),
        dict(m=65,e=2,q=1,k=32,n=65,route="single",style="random",seed=6),
        dict(m=97,e=1,q=1,k=65,n=96,route="single",style="marker",seed=7),
        dict(m=113,e=1,q=1,k=129,n=129,route="single",style="random",seed=8),
        dict(m=16,e=8,q=4,k=17,n=17,route="balanced",style="random",seed=9),
        dict(m=17,e=5,q=2,k=65,n=65,route="balanced",style="random",seed=10),
        dict(m=33,e=8,q=3,k=64,n=64,route="skew",style="random",seed=11),
        dict(m=129,e=8,q=4,k=129,n=129,route="random",style="random",seed=12),
        dict(m=129,e=8,q=4,k=129,n=129,route="balanced",style="positive",seed=13),
        dict(m=67,e=7,q=3,k=127,n=97,route="random",style="cancel",seed=14),
        dict(m=33,e=8,q=2,k=33,n=65,route="skew",style="zero",seed=15),
        dict(m=5,e=5,q=1,k=1,n=65,route="balanced",style="marker",seed=16),
        dict(m=31,e=4,q=4,k=129,n=33,route="balanced",style="subnormal",seed=17),
        dict(m=97,e=8,q=2,k=96,n=128,route="random",style="random",seed=18),
        dict(m=129,e=8,q=1,k=128,n=127,route="skew",style="random",seed=19),
        dict(m=64,e=4,q=1,k=32,n=64,route="balanced",style="marker",seed=20),
    ]


def arrays(plan):
    m,e,q,k,n = [plan[key] for key in ("m","e","q","k","n")]
    rng = np.random.default_rng(plan["seed"])
    x = rng.uniform(-.5,.5,(m,k)).astype(np.float16)
    weights = rng.uniform(-.5,.5,(e,k,n)).astype(np.float16)
    if plan["route"] == "single":
        ids = np.tile(np.arange(q,dtype=np.int32),(m,1))
    elif plan["route"] == "balanced":
        ids = ((np.arange(m)[:,None]+np.arange(q)[None,:])%e).astype(np.int32)
    elif plan["route"] == "skew":
        ids = np.tile(np.arange(q,dtype=np.int32),(m,1))
        ids[-1] = np.arange(e-q,e,dtype=np.int32)
    else:
        ids = np.argsort(rng.random((m,e)),axis=1)[:,:q].astype(np.int32).copy()
    route = rng.uniform(.1,1,(m,q))
    route = (route/(route.sum(axis=1,keepdims=True)+.25)).astype(np.float32)
    style = plan["style"]
    if style in {"positive","marker"}:
        x.fill(.5)
        weights.fill(.5)
        route.fill(1/q if q in (1,2,4) else .25)
    if style == "marker":
        for expert in range(e):
            weights[expert] = np.float16((expert+1)/(2*e))
    elif style == "cancel":
        x.fill(.5)
        weights[:] = np.where(np.arange(k)[None,:,None]%2,.5,-.5)
    elif style == "zero":
        route.fill(0)
    elif style == "subnormal":
        x[:] = (rng.integers(-1023,1024,size=x.shape)*2**-24).astype(np.float16)
        weights.fill(.5)
    return x,weights,ids,route


def independent_reference(values):
    """NumPy FP64 expert-major calculation, no sorting/padding/tile decoder."""
    x,weights,ids,route = values
    out = np.zeros((x.shape[0],weights.shape[-1]),dtype=np.float64)
    for expert in range(weights.shape[0]):
        token,choice = np.where(ids == expert)
        out[token] += route[token,choice,None].astype(np.float64)*np.einsum('tk,kn->tn',x[token].astype(np.float64),weights[expert].astype(np.float64),optimize=False)
    return out


def route_layout(ids,experts):
    flat = ids.reshape(-1)
    routes=[]
    block_experts=[]
    for expert in range(experts):
        found = np.flatnonzero(flat == expert).tolist()
        padded = ((len(found)+15)//16)*16
        routes.extend(found+[flat.size]*(padded-len(found)))
        block_experts.extend([expert]*(padded//16))
    return np.array(routes,dtype=np.int32),np.array(block_experts,dtype=np.int32)


def schedule(rows,columns,mutation=False):
    group = 4 if rows>=4 else 1
    coords=[]
    for pid in range(rows*columns):
        first = (pid//(group*columns))*group
        height = min(rows-first,group)
        local = pid%(group*columns)
        coords.append((first+local%height,local//(group if mutation else height)))
    return coords


def scheduled_reference(values,mutation):
    x,weights,ids,route=values
    sorted_routes,experts=route_layout(ids,weights.shape[0])
    routed=np.zeros((ids.size,weights.shape[2]),np.float64)
    for row,col in schedule(len(experts),(weights.shape[2]+31)//32,mutation):
        tokens=sorted_routes[row*16:(row+1)*16]
        tokens=tokens[tokens<ids.size]
        columns=slice(col*32,min((col+1)*32,weights.shape[2]))
        routed[tokens,columns]=np.einsum('tk,kn->tn',x[tokens//ids.shape[1]].astype(np.float64),weights[experts[row],:,columns].astype(np.float64),optimize=False)*route.reshape(-1)[tokens,None]
    return routed.reshape(ids.shape+(weights.shape[2],)).sum(axis=1)


def _public_cpu_namespace(case_root):
    source=(case_root/'case_45'/'kernel.py').read_text()
    names={'validate_inputs','reference','_prepare_routes','make_inputs'}
    nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in names]
    namespace={'torch':torch}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'moe_cpu_public','exec'),namespace)
    return namespace


def cpu_audit(case_root):
    public=_public_cpu_namespace(case_root)
    # Route count <=516, E<=8; padding adds <=8*15, hence <=40 row blocks.
    schedule_rows=[]
    for rows in range(1,41):
        for columns in range(1,6):
            coords=schedule(rows,columns)
            expected={(r,c) for r in range(rows) for c in range(columns)}
            assert len(coords)==len(set(coords)) and set(coords)==expected
            broken=schedule(rows,columns,True)
            missing=sorted(expected-set(broken))
            duplicates=sum(v-1 for v in Counter(broken).values() if v>1)
            schedule_rows.append(dict(row_blocks=rows,column_blocks=columns,
                correct_bijective=True,mutation_missing=len(missing),mutation_duplicate=duplicates,
                first_missing=missing[0] if missing else None))
    rows=[]
    for plan in plans():
        values=arrays(plan)
        host=[torch.from_numpy(v.copy()) for v in values]
        public['validate_inputs'](*host,require_cuda=False)
        oracle=independent_reference(values)
        reference=public['reference'](*host).numpy()
        layout=route_layout(values[2],values[1].shape[0])
        prepared=public['_prepare_routes'](host[2],values[1].shape[0],16)
        assert all(np.array_equal(a,b.numpy()) for a,b in zip(layout,prepared))
        scheduled=scheduled_reference(values,False)
        broken=scheduled_reference(values,True)
        referr=float(np.max(np.abs(oracle-reference)))
        gooderr=float(np.max(np.abs(oracle-scheduled)))
        assert referr<1e-10 and gooderr<1e-10
        rows.append(dict(plan,public_reference_max_error=referr,correct_schedule_max_error=gooderr,
            mutation_max_error=float(np.max(np.abs(oracle-broken))),row_blocks=len(layout[1])))
    invalid=[]
    for kind in ('bounds','duplicate_route','route_sum','negative_view','grad','inference_tensor','ids_dtype','overlap'):
        values=[torch.from_numpy(v.copy()) for v in arrays(plans()[9])]
        if kind=='bounds': values[0][0,0]=1
        elif kind=='duplicate_route': values[2][0,1]=values[2][0,0]
        elif kind=='route_sum': values[3].fill_(1)
        elif kind=='negative_view': values[0]=torch._neg_view(values[0])
        elif kind=='grad': values[0].requires_grad_(True)
        elif kind=='inference_tensor':
            with torch.inference_mode(): values[0]=values[0].clone()
        elif kind=='ids_dtype': values[2]=values[2].long()
        elif kind=='overlap':
            values[0]=values[1].reshape(-1)[:values[0].numel()].reshape(values[0].shape)
        try: public['validate_inputs'](*values,require_cuda=False)
        except ValueError as exc: invalid.append({'kind':kind,'correctly_rejected':True,'reason':str(exc)})
        else: raise AssertionError(f'Validator accepted {kind}')
    return {'passed':True,'schedule_domain_checks':schedule_rows,'plans':rows,'invalid_inputs':invalid,
            'method':'Exhaustive bounded tile-grid bijection; NumPy expert-major FP64 vs public token-major reference; actual public route preparation vs independent expert grouping.'}


def _load(case_root,name):
    key='moe_qualification_'+name
    spec=importlib.util.spec_from_file_location(key,case_root/name/'kernel.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[key]=module
    spec.loader.exec_module(module)
    return module


def validate(case_root):
    cpu=cpu_audit(case_root)
    cases={}
    for name,label in EXPECTED.items():
        module=_load(case_root,name)
        rows=[]
        for plan in plans():
            values=arrays(plan)
            ref=independent_reference(values)
            base=[torch.from_numpy(v.copy()).cuda() for v in values]
            # Keep every transformation, including the original, at fixed weights.
            variations=[('base',base)]
            if plan['q']>1:
                variations.append(('choice_permutation',[base[0],base[1],base[2].flip(1).contiguous(),base[3].flip(1).contiguous()]))
            if plan['e']>1:
                perm=torch.arange(plan['e']-1,-1,-1,device='cuda')
                variations.append(('expert_relabel',[base[0],base[1][perm].contiguous(),(plan['e']-1-base[2]).int(),base[3]]))
            if plan['seed']==10:
                offset_inputs=[]
                for tensor in base:
                    storage=torch.empty(tensor.numel()+13,device=tensor.device,dtype=tensor.dtype)
                    view=storage[7:7+tensor.numel()].view(tensor.shape)
                    view.copy_(tensor)
                    offset_inputs.append(view)
                variations.append(('nonzero_storage_offset',offset_inputs))
            for variation,gpu in variations:
                legality=module.validate_inputs(*gpu)
                saved=[v.clone() for v in gpu]
                actual=module.run(*gpu)
                torch.cuda.synchronize()
                actual_np=actual.double().cpu().numpy()
                delta=np.abs(actual_np-ref)
                ratio=delta/(.01+.0002*np.abs(ref))
                finite=bool(np.isfinite(actual_np).all())
                preserved=all(torch.equal(a,b) for a,b in zip(gpu,saved))
                types_valid=actual.dtype==torch.float32 and actual.device==gpu[0].device and tuple(actual.shape)==tuple(ref.shape)
                output_range=(actual.data_ptr(),actual.data_ptr()+actual.numel()*actual.element_size())
                fresh=all(output_range[1]<=v.data_ptr() or v.data_ptr()+v.numel()*v.element_size()<=output_range[0] for v in gpu)
                first=None
                if not finite or float(ratio.max())>1:
                    idx=np.unravel_index(np.argmax(np.nan_to_num(ratio,nan=np.inf)),ratio.shape)
                    first={'index':list(map(int,idx)),'actual':float(actual_np[idx]),'reference':float(ref[idx]),
                           'absolute_error':float(delta[idx]),'allowed_error':float(.01+.0002*abs(ref[idx])),
                           'ratio':float(ratio[idx])}
                rows.append(dict(plan,variation=variation,legal=legality['valid'],passed=bool(finite and preserved and types_valid and fresh and ratio.max()<=1),
                    max_error_ratio=float(ratio.max()),max_abs_error=float(delta.max()),finite=finite,
                    inputs_preserved=preserved,shape_dtype_valid=types_valid,output_fresh=fresh,witness=first))
        failures=[r for r in rows if not r['passed']]
        qualified=not failures if label=='trust' else bool(failures)
        cases[name]={'ground_truth':label,'qualification_passed':qualified,'tests':rows,
            'tests_total':len(rows),'tests_passed':len(rows)-len(failures),'counterexamples':failures,
            'label_scope':'Fixed implementation over the complete public contract; input samples do not define labels.',
            'analysis':'Independent route layout, complete bounded tile-grid bijection and FP64 numerical oracle; explicit zero-initialized route buffer makes missing-output witnesses deterministic. Finite numerical testing is not a formal proof.'}
    return {'cases':cases,'cpu_audit':cpu,'qualification_passed':all(v['qualification_passed'] for v in cases.values())}
