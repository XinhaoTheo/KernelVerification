
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_16/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = {}
for (M,N,K) in [(64,64,40),(100,70,100),(64,64,64),(70,100,128),(32,32,16)]:
    a = torch.randn(M,K,device='cuda',dtype=torch.float32)
    b = torch.randn(K,N,device='cuda',dtype=torch.float32)
    c = m.matmul(a,b)
    ref = a@b
    Kt = (K//32)*32
    trunc = a[:,:Kt]@b[:Kt,:]
    res[f"M{M}_N{N}_K{K}"] = {
        "max_abs_err_vs_full": float((c-ref).abs().max()),
        "max_abs_err_vs_truncated": float((c-trunc).abs().max()),
        "ref_absmax": float(ref.abs().max()),
        "Ktrunc": Kt,
    }
print(json.dumps(res, indent=1))
