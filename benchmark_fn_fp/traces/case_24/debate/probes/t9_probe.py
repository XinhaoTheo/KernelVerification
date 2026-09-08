
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_24/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(1)
dev='cuda'
res=[]
for (M,N) in [(128,4096),(64,8192),(32,1000),(16,257),(8,1),(8,255),(8,256),(64,4095)]:
    A=torch.randn(M,N,device=dev,dtype=torch.float32)
    B=torch.randn(M,N,device=dev,dtype=torch.float32)
    out=k.cosine_similarity(A,B).double()
    Ad=A.double(); Bd=B.double()
    ref=(Ad*Bd).sum(1)/(Ad.norm(dim=1)*Bd.norm(dim=1))
    err=(out-ref).abs()
    near = ref.abs()<1e-2
    # also a correlated case (cos near 1)
    B2 = A + 0.001*torch.randn_like(A)
    out2=k.cosine_similarity(A,B2).double()
    ref2=(Ad*B2.double()).sum(1)/(Ad.norm(dim=1)*B2.double().norm(dim=1))
    res.append(dict(shape=[M,N], N_mod_256=N%256,
        max_abs_err=float(err.max()),
        max_abs_err_near_orthogonal=float(err[near].max()) if near.any() else None,
        n_near_orthogonal=int(near.sum()),
        max_abs_err_dominant_cos_near1=float((out2-ref2).abs().max()),
        max_out_abs=float(out.abs().max()),
        max_out2=float(out2.max()),
        any_nan=bool(torch.isnan(out).any() or torch.isnan(out2).any())))
print(json.dumps(dict(cases=res, overall_max_abs_err=max(r["max_abs_err"] for r in res),
   overall_max_abs_err_dominant=max(r["max_abs_err_dominant_cos_near1"] for r in res))))
