
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
dev='cuda'
res={"trials":[], "violations":0, "mismatch":0, "total_rows":0}
def ref(t,d,inv_q):
    p=torch.clamp(t-d,min=0)
    s=p*inv_q
    return torch.argmax(s,dim=1)
configs=[]
for V in [16,128,1000,4096]:
    for temp in [0.3,1.0,5.0]:
        configs.append((V,temp))
for V,temp in configs:
    B=64
    t=torch.softmax(torch.randn(B,V,device=dev)/temp,dim=1)
    d=torch.softmax(torch.randn(B,V,device=dev)/temp,dim=1)
    q=torch.empty(B,V,device=dev).exponential_(1.0)
    inv_q=1.0/q
    out=k.sample_recovered_tokens(t,d,inv_q)
    r=ref(t,d,inv_q)
    mism=(out!=r)
    resid=(t-d).gather(1,out.view(-1,1)).squeeze(1)
    viol=(resid<0)
    res["total_rows"]+=B
    res["mismatch"]+=int(mism.sum())
    res["violations"]+=int(viol.sum())
    if mism.any():
        i=int(mism.nonzero()[0])
        sk=((t-d)*inv_q)[i,out[i]].item(); sr=(torch.clamp(t-d,min=0)*inv_q)[i,r[i]].item()
        res["trials"].append({"V":V,"temp":temp,"row":i,"kernel_tok":int(out[i]),"ref_tok":int(r[i]),
                              "kernel_resid":float((t-d)[i,out[i]]),"ref_resid":float((t-d)[i,r[i]]),
                              "kernel_score":sk,"ref_score":sr})
# adversarial: draft dominates target almost everywhere (still proper distributions)
B,V=32,512
base=torch.softmax(torch.randn(B,V,device=dev),dim=1)
t=base.clone()
d=base.clone()
# move mass: make d nearly equal t but slightly larger on all but one token
eps=1e-7
d=d*(1+eps); d[:,0]-= (d.sum(1)-1.0)  # renormalize by adjusting token 0
d=d.clamp(min=0); d=d/d.sum(1,keepdim=True)
q=torch.empty(B,V,device=dev).exponential_(1.0); inv_q=1.0/q
out=k.sample_recovered_tokens(t,d,inv_q); r=ref(t,d,inv_q)
res["adv_nearequal"]={"mismatch":int((out!=r).sum()),
                      "neg_resid_returned":int(((t-d).gather(1,out.view(-1,1)).squeeze(1)<0).sum()),
                      "rows_with_no_positive_residual":int((((t-d)>0).sum(1)==0).sum())}
# identical distributions (all residuals exactly zero)
t2=base.clone(); d2=base.clone()
out2=k.sample_recovered_tokens(t2,d2,inv_q); r2=ref(t2,d2,inv_q)
res["identical"]={"kernel_first5":out2[:5].tolist(),"ref_first5":r2[:5].tolist(),"mismatch":int((out2!=r2).sum())}
print(json.dumps(res,indent=1))
