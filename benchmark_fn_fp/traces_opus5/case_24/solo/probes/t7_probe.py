
import torch, json, importlib.util, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_24/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref(a,b,eps=1e-8):
    A=a.double(); B=b.double()
    dot=(A*B).sum(1); na=A.pow(2).sum(1).sqrt(); nb=B.pow(2).sum(1).sqrt()
    den=(na*nb).clamp_min(eps)
    return dot/den

res={}
torch.manual_seed(0)
for (R,N) in [(4,16),(8,256),(8,257),(16,1000),(32,4096),(4,100000),(3,1)]:
    a=torch.randn(R,N,device='cuda',dtype=torch.float32)
    b=torch.randn(R,N,device='cuda',dtype=torch.float32)
    o=k.cosine_similarity(a,b)
    r=ref(a,b)
    res[f"{R}x{N}"]={"maxabs":float((o.double()-r).abs().max()),
                     "out_range":[float(o.min()),float(o.max())],
                     "finite":bool(torch.isfinite(o).all())}

# scaled magnitudes
for scale in [1e-4, 1e4, 1e-20, 1e20]:
    a=(torch.randn(8,512,device='cuda')*scale)
    b=(torch.randn(8,512,device='cuda')*scale)
    o=k.cosine_similarity(a,b); r=ref(a,b)
    res[f"scale{scale:g}"]={"maxabs":float((o.double()-r).abs().max()),
                            "finite":bool(torch.isfinite(o).all()),
                            "out":[round(float(x),6) for x in o[:4]],
                            "ref":[round(float(x),6) for x in r[:4]]}

# highly correlated / near-identical rows (should be ~1)
a=torch.randn(8,4096,device='cuda')
b=a+1e-6*torch.randn_like(a)
o=k.cosine_similarity(a,b); r=ref(a,b)
res["near_identical"]={"maxabs":float((o.double()-r).abs().max()),"out":[float(x) for x in o[:3]]}

print(json.dumps(res,indent=1))
