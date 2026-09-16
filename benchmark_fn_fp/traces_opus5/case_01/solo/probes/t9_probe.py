
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_01/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(1)
dev='cuda'
def ref(t,d,inv_q):
    return torch.argmax(torch.clamp(t-d,min=0)*inv_q, dim=1)
out_res={}

def check(name,t,d,inv_q):
    o=k.sample_recovered_tokens(t,d,inv_q)
    r=ref(t,d,inv_q)
    resid=(t-d).gather(1,o.view(-1,1)).squeeze(1)
    npos=((t-d)>0).sum(1)
    mm=(o!=r)
    entry={"rows":int(t.shape[0]),"mismatch":int(mm.sum()),
           "neg_resid_returned":int((resid<0).sum()),
           "rows_no_positive_resid":int((npos==0).sum()),
           "min_row_sum_t":float(t.sum(1).min()),"min_row_sum_d":float(d.sum(1).min())}
    if mm.any():
        i=int(mm.nonzero()[0])
        entry["example"]={"row":i,"kernel_tok":int(o[i]),"ref_tok":int(r[i]),
            "kernel_resid":float((t-d)[i,o[i]]),"ref_resid":float((t-d)[i,r[i]]),
            "kernel_score_unclamped":float(((t-d)*inv_q)[i,o[i]]),
            "ref_score_clamped":float((torch.clamp(t-d,min=0)*inv_q)[i,r[i]]),
            "npos_in_row":int(npos[i])}
    out_res[name]=entry

B,V=256,1024
# A: draft extremely peaked on token 0, target uniform -> token 0 residual hugely negative
t=torch.full((B,V),1.0/V,device=dev)
d=torch.full((B,V),0.01/(V-1),device=dev); d[:,0]=0.99
q=torch.empty(B,V,device=dev).exponential_(1.0); inv_q=1.0/q
inv_q[:,0]=1e6   # enormous inv_q exactly on the forbidden token
check("A_peaked_draft_huge_invq_on_neg", t,d,inv_q)

# B: many strongly negative residuals with large inv_q, few tiny positives
t=torch.softmax(torch.randn(B,V,device=dev)*3,dim=1)
d=torch.softmax(torch.randn(B,V,device=dev)*3,dim=1)
inv_q=1.0/torch.empty(B,V,device=dev).exponential_(1.0)
neg=(t-d)<0
inv_q=torch.where(neg, inv_q*1e5, inv_q*1e-3)  # bias race hugely toward negative-residual tokens
check("B_invq_biased_to_negatives", t,d,inv_q)

# C: target == draft except one token pair (minimal positive residual), inv_q huge elsewhere
base=torch.softmax(torch.randn(B,V,device=dev),dim=1)
t=base.clone(); d=base.clone()
idx=torch.randint(1,V,(B,),device=dev)
delta=1e-6
t.scatter_add_(1, idx.view(-1,1), torch.full((B,1),delta,device=dev))
t[:,0]-=delta
inv_q=torch.full((B,V),1e6,device=dev); inv_q.scatter_(1, idx.view(-1,1), torch.ones(B,1,device=dev))
check("C_single_tiny_positive_residual", t,d,inv_q)

# D: exact-tie stress: several tokens with identical max score
t=torch.full((B,V),1.0/V,device=dev)
d=torch.full((B,V),1.0/V,device=dev)
d[:,:8]=1.0/V - 1e-4
t[:,8:16]=1.0/V - 1e-4
inv_q=torch.ones(B,V,device=dev)
check("D_ties", t,d,inv_q)
print(json.dumps(out_res))
