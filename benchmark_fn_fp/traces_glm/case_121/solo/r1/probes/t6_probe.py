import torch, itertools, random, json
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_121/kernel.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
dev="cuda"
def gen(seed, m,k,n,gs):
    rng = torch.Generator("cpu").manual_seed(seed)
    x = (torch.rand((m,k),generator=rng)-.5).half()
    codes = torch.randint(-2**31,2**31,(((k+7)//8),n),generator=rng,dtype=torch.int64).clamp(-2**31,2**31-1)
    qweight = codes.to(torch.int32)
    G = (k+gs-1)//gs
    zc = torch.randint(-2**31,2**31,(G,(n+7)//8),generator=rng,dtype=torch.int64).clamp(-2**31,2**31-1)
    qzeros = zc.to(torch.int32)
    scales = (.002+.029*torch.rand((G,n),generator=rng)).half()
    # valid g_idx histogram, permuted
    ids = torch.cat([torch.full((min(gs,k-g*gs),),g) for g in range(G)])
    ids = ids[torch.randperm(len(ids),generator=rng)].int()
    return (x.to(dev),qweight.to(dev),scales.to(dev),qzeros.to(dev),ids.to(dev),gs)
worst=0.0; worst_case=None; fails=0; tests=0
random.seed(0)
cases=[]
for gs in (16,32,64):
    for k in (1,15,16,17,31,33,63,64,65,96,127,128):
        cases.append((random.randint(1,65),k,random.randint(1,96),gs))
for seed,(m,k,n,gs) in enumerate(cases):
    x,qw,sc,qz,ids,gs2 = gen(seed+100,m,k,n,gs)
    out = K.run(x,qw,sc,qz,ids,gs2)
    ref = K.reference(x,qw,sc,qz,ids,gs2)
    tol = 0.01 + 0.001*ref.abs()
    err = (out.double()-ref.double()).abs()
    viol = (err > tol)
    tests+=1
    e=err.max().item()
    rel=(err/tol).max().item()
    if rel>worst: worst=rel; worst_case=(m,k,n,gs,seed)
    if viol.any():
        fails+=1
        print("FAIL", m,k,n,gs,"maxerr",e,"numviol",int(viol.sum()))
print(json.dumps({"tests":tests,"fails":fails,"worst_err_over_tol":worst,"worst_case":worst_case}))