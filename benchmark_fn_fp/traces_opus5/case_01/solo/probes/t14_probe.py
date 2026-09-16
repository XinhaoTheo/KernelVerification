
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(7)
dev='cuda'
res={}
def ref(t,d,inv_q):
    return torch.argmax(torch.clamp(t-d,min=0)*inv_q, dim=1)

for V in [32000, 128256]:
    B=16
    base=torch.softmax(torch.randn(B,V,device=dev),dim=1)
    t=base.clone(); d=base.clone()
    # give draft slightly more mass everywhere except one token -> exactly one positive residual
    idx=torch.randint(0,V,(B,),device=dev)
    delta=torch.full((B,1),1e-7,device=dev)
    t.scatter_add_(1, idx.view(-1,1), delta)
    # remove that mass from t spread over all others: subtract tiny uniform amount
    t = t - (delta/ V)
    t.scatter_add_(1, idx.view(-1,1), delta/V)
    t = t.clamp(min=0)
    # inv_q: huge on every token except the positive-residual one
    inv_q=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
    inv_q = inv_q*1e6
    inv_q.scatter_(1, idx.view(-1,1), torch.full((B,1),1e-6,device=dev))
    o=k.sample_recovered_tokens(t,d,inv_q)
    r=ref(t,d,inv_q)
    resid=(t-d)
    npos=(resid>0).sum(1)
    sel=resid.gather(1,o.view(-1,1)).squeeze(1)
    entry={"V":V,"rows":B,"mismatch":int((o!=r).sum()),
           "neg_resid_returned":int((sel<0).sum()),
           "rows_no_positive_resid":int((npos==0).sum()),
           "median_npos":int(npos.median()),
           "row_sum_t_range":[float(t.sum(1).min()),float(t.sum(1).max())],
           "row_sum_d_range":[float(d.sum(1).min()),float(d.sum(1).max())],
           "kernel_tok0":int(o[0]),"ref_tok0":int(r[0]),"idx0":int(idx[0])}
    res["single_tiny_pos_V%d"%V]=entry

# also a fully realistic case at V=128256 with plain exponential inv_q, larger batch
for V in [32000,128256]:
    B=64
    t=torch.softmax(torch.randn(B,V,device=dev)*1.5,dim=1)
    d=torch.softmax(torch.randn(B,V,device=dev)*1.5,dim=1)
    inv_q=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
    o=k.sample_recovered_tokens(t,d,inv_q)
    r=ref(t,d,inv_q)
    sel=(t-d).gather(1,o.view(-1,1)).squeeze(1)
    res["realistic_V%d"%V]={"rows":B,"mismatch":int((o!=r).sum()),
        "neg_resid_returned":int((sel<0).sum()),
        "dtype_out":str(o.dtype),"shape_out":list(o.shape)}
print(json.dumps(res))
