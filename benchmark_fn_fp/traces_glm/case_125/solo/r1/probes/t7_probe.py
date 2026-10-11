import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_125/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"
C=32
shifts=torch.arange(0,32,dtype=torch.int32)
g=torch.Generator("cpu").manual_seed(1)
mult=torch.randint(2**30,2**31,(C,),dtype=torch.int64,generator=g).to(torch.int32)
zeros=torch.tensor([0,127,-128]*11,dtype=torch.int32)[:C]
exps=torch.tensor([-8,-7,-6,-5,-4,-3,-2,-1,0]*4,dtype=torch.int32)[:C]
scales=torch.pow(torch.tensor(2.0),exps.to(torch.float64)).to(torch.float32)
m64=mult.to(torch.int64)

# Big pool of candidate accumulators (vectorized) — 300k random int32
pool=torch.randint(-(2**31),2**31,(300000,C),dtype=torch.int64,generator=torch.Generator("cpu").manual_seed(3))
P=pool*m64
H=(P+(1<<30))//((1<<31))

rows=[(-2**31)*torch.ones(C,dtype=torch.int64),(2**31-1)*torch.ones(C,dtype=torch.int64),torch.zeros(C,dtype=torch.int64)]
# R-stage exact ties: positive and negative, per channel
for sign in (+1,-1):
    tie=torch.zeros(C,dtype=torch.int64)
    for r in range(1,32):
        cand=pool[(H.sign()==sign)&(H.abs()%(1<<r)==(1<<(r-1)))]
        if cand.numel(): tie[r]=cand[0,r]
    rows.append(tie)
# H-stage exact ties (P+2^30 divisible by 2^31) pos/neg
for sign in (+1,-1):
    tie=torch.zeros(C,dtype=torch.int64)
    cand=pool[(P%(1<<31)==((-(1<<30))%(1<<31)))&(H.sign()==sign)&(H.abs()>0)]
    if cand.numel(): tie=cand[0]
    rows.append(tie)
# saturation-boundary H values
for target in (127,128,129,-127,-128,-129,255,256,-255,-256):
    row=torch.zeros(C,dtype=torch.int64)
    cand=pool[(H==target)]
    if cand.numel(): row=cand[0]
    rows.append(row)
# random block
rows.append(pool[:200])
acc=torch.stack([r if r.dim()==1 else r[0] for r in rows]) if False else None
mat=torch.cat([r.reshape(-1,C) for r in rows],0)[:,:C]
acc=mat.to(torch.int32).to(dev)
params=tuple(t.to(dev) for t in (mult,shifts,zeros,scales))
out=k.run(acc,*params)
ref=k.reference(acc,*params)
diff=(out.double()-ref.double()).abs()
mism=(diff>1e-6).sum().item()
info={"shape":list(acc.shape),"max_abs_err":diff.max().item(),"mismatches":mism}
if mism:
    idx=(diff>1e-6).nonzero()[:10]
    for i,j in idx.tolist():
        info[f"mm{i}_{j}"]={"a":int(acc[i,j]),"m":int(mult[j]),"r":int(shifts[j]),"z":int(zeros[j]),"got":out[i,j].item(),"ref":ref[i,j].item()}
print(json.dumps(info))
