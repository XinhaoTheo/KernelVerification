
import json, importlib.util, torch, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_24/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
dev='cuda'
N=1024
rows=[]
labels=[]
scales=[1.0, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7]
base = torch.randn(len(scales), N, device=dev, dtype=torch.float64).abs()+0.5
A=torch.empty(len(scales),N,device=dev,dtype=torch.float32)
B=torch.empty_like(A)
for i,s in enumerate(scales):
    A[i]=(base[i]*s).float()
    B[i]=(base[i]*s).float()   # identical -> true cos = 1
# add an all-zero row and a mixed row (one tiny norm, one unit norm)
extra_a = torch.zeros(2,N,device=dev,dtype=torch.float32)
extra_b = torch.zeros(2,N,device=dev,dtype=torch.float32)
extra_a[1]=(base[0]*1e-9).float(); extra_b[1]=base[0].float()
A=torch.cat([A,extra_a],0); B=torch.cat([B,extra_b],0)
labels=[f"scale={s}" for s in scales]+["all_zero_row","tiny_a_unit_b"]
out=k.cosine_similarity(A,B).double()
Ad=A.double(); Bd=B.double()
na=Ad.norm(dim=1); nb=Bd.norm(dim=1); dot=(Ad*Bd).sum(1)
exact=torch.where(na*nb>0, dot/(na*nb), torch.zeros_like(dot))
tref=torch.nn.functional.cosine_similarity(A,B,dim=1,eps=1e-8).double()
prodfloor = dot/torch.clamp(na*nb, min=1e-8)
recs=[]
for i,l in enumerate(labels):
    recs.append(dict(label=l, na=float(na[i]), nb=float(nb[i]), prod=float(na[i]*nb[i]),
                     kernel=float(out[i]), exact_cos=float(exact[i]),
                     torch_F=float(tref[i]), literal_product_floor=float(prodfloor[i]),
                     abs_err_vs_exact=float(abs(out[i]-exact[i])),
                     abs_err_vs_torchF=float(abs(out[i]-tref[i])),
                     abs_err_vs_literal=float(abs(out[i]-prodfloor[i]))))
nz = [r for r in recs if r["label"] not in ("all_zero_row",)]
print(json.dumps(dict(rows=recs,
  max_abs_err_vs_exact_nonzero=max(r["abs_err_vs_exact"] for r in nz),
  max_abs_err_vs_torchF_nonzero=max(r["abs_err_vs_torchF"] for r in nz),
  max_abs_err_vs_literal_productfloor=max(r["abs_err_vs_literal"] for r in recs)), indent=None))
