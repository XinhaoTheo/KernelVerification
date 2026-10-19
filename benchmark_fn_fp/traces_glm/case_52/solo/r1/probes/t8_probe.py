import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_125/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev="cuda"; C=32
shifts=torch.arange(0,32,dtype=torch.int32)
g=torch.Generator("cpu").manual_seed(1)
mult=torch.randint(2**30,2**31,(C,),dtype=torch.int64,generator=g).to(torch.int32)
zeros=torch.tensor([0,127,-128]*11,dtype=torch.int32)[:C]
exps=torch.tensor([-8,-7,-6,-5,-4,-3,-2,-1,0]*4,dtype=torch.int32)[:C]
scales=torch.pow(torch.tensor(2.0),exps.to(torch.float64)).to(torch.float32)
m64=mult.to(torch.int64)

pool=torch.randint(-(2**31),2**31,(400000,C),dtype=torch.int64,generator=torch.Generator("cpu").manual_seed(3))
P=pool*m64
H=(P+(1<<30))//(1<<31)

rows=[(-2**31)*torch.ones(C,dtype=torch.int64),(2**31-1)*torch.ones(C,dtype=torch.int64),torch.zeros(C,dtype=torch.int64)]
# R-stage exact ties per channel (column r, both signs)
Hmask=pool.view(-1)  # placeholder
for r in range(1,32):
    tie=torch.zeros(C,dtype=torch.int64)
    col=H[:,r]
    for sign in (+1,-1):
        m=(col.sign()==sign)&(col.abs()%(1<<r)==(1<<(r-1)))
        if m.any(): tie[r]=pool[:,r][m][0]
    rows.append(tie)
# H-stage exact ties (per channel, P+2^30 divisible by 2^31)
for r in range(C):
    tie=torch.zeros(C,dtype=torch.int64)
    m=((P[:,r]+(1<<30))%(1<<31)==0)&(H[:,r]!=0)
    if m.any(): tie=pool[m][0]
    rows.append(tie)
# saturation boundary H values
for target in (127,128,129,-127,-128,-129,255,256,-255,-256):
    row=torch.zeros(C,dtype=torch.int64)
    m=(H[:,0]==target)
    if m.any():
        row=pool[m][0]
        # remap to channel-specific: find per-channel candidates
        row=torch.zeros(C,dtype=torch.int64)
        for c in range(C):
            mc=(H[:,c]==target)
            if mc.any(): row[c]=pool[:,c][mc][0]
    rows.append(row)
rows.append(pool[:200])
mat=torch.cat([r.reshape(-1,C) for r in rows],0)
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
        info[f"mm{i}_{j}"]={"a":int(acc[i,j]),"m":int(mult[j]),"r":int(shifts[j]),"z":int(zeros[j]),"s":float(scales[j]),"got":out[i,j].item(),"ref":ref[i,j].item()}
print(json.dumps(info))
