
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_24/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(2)
dev='cuda'
out={}
# 1) For standard-benchmark-style randn inputs, how far is the norm product from eps=1e-8?
mins=[]
for (M,N) in [(128,4096),(64,8192),(32,1000),(256,512),(16,257)]:
    A=torch.randn(M,N,device=dev); B=torch.randn(M,N,device=dev)
    prod=(A.double().norm(dim=1)*B.double().norm(dim=1))
    mins.append(dict(shape=[M,N], min_norm_product=float(prod.min()),
                     ratio_to_eps=float(prod.min()/1e-8)))
out['randn_shapes']=mins
# 2) Threshold sweep: uniform scale s applied to identical rows, N=1024, find where abs err exceeds 1e-3
N=1024
base=(torch.randn(1,N,device=dev,dtype=torch.float64).abs()+0.5)
sweep=[]
first_bad=None
for e in range(0,13):
    s=10.0**(-e)
    a=(base*s).float(); b=a.clone()
    kv=float(k.cosine_similarity(a,b)[0])
    ad=a.double(); na=ad.norm(); prod=float(na*na)
    exact=float((ad*ad).sum()/(na*na)) if na>0 else 0.0
    err=abs(kv-exact)
    sweep.append(dict(entry_scale=s, norm_product=prod, kernel=kv, exact=exact, abs_err=err))
    if first_bad is None and err>1e-3:
        first_bad=dict(entry_scale=s, norm_product=prod, abs_err=err)
out['scale_sweep']=sweep
out['first_scale_with_abs_err_gt_1e-3']=first_bad
# 3) Does the failure require BOTH norms small? one small one unit
recs=[]
for e in [3,5,6,7,9]:
    s=10.0**(-e)
    a=(base*s).float(); b=base.float()
    kv=float(k.cosine_similarity(a,b)[0])
    ad=a.double(); bd=b.double()
    exact=float((ad*bd).sum()/(ad.norm()*bd.norm()))
    recs.append(dict(a_scale=s, norm_product=float(ad.norm()*bd.norm()),
                     kernel=kv, exact=exact, abs_err=abs(kv-exact)))
out['asymmetric']=recs
print(json.dumps(out))
