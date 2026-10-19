import torch, sys, json
sys.path.insert(0, '/root/cases/case_160')
import kernel as K
dev='cuda'
# extreme cases: all -128, all +127, mixed max-magnitude
cases={}
for name,(tv,fv) in {"all_neg128":(-128,-128),"all_pos127":(127,127),"tile128_filt127":(-128,127),"tile127_filt_neg128":(127,-128)}.items():
    t=torch.full((64,6,6),tv,dtype=torch.int8)
    f=torch.full((64,3,3),fv,dtype=torch.int8)
    out=K.run(t.to(dev),f.to(dev))
    ref=K.reference(t,f).to(dev)
    cases[name]={"mismatch_count":int((out!=ref).sum()),"total":out.numel(),
                 "kernel_sample":out[0,0,:3].tolist(),"ref_sample":ref[0,0,:3].tolist()}
# also random extreme-magnitude (values in {-128,127})
gen=torch.Generator('cpu').manual_seed(7)
t=torch.tensor(torch.randint(0,2,(256,6,6),generator=gen).numpy()*255-128,dtype=torch.int8)
f=torch.tensor(torch.randint(0,2,(256,3,3),generator=gen).numpy()*255-128,dtype=torch.int8)
out=K.run(t.to(dev),f.to(dev)); ref=K.reference(t,f).to(dev)
cases["random_pm"]={"mismatch_count":int((out!=ref).sum()),"total":out.numel()}
print(json.dumps(cases))
