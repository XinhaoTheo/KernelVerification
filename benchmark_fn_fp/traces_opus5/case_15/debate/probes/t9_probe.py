
import json, sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_16/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = {}
for K in [32, 64, 128, 33, 40, 47, 100]:
    M = N = 64
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    c = m.matmul(a, b)
    ref = (a.double() @ b.double()).float()
    Kt = (K // 32) * 32
    trunc = (a[:, :Kt].double() @ b[:Kt].double()).float()
    err_full = (c - ref).abs().max().item()
    err_trunc = (c - trunc).abs().max().item()
    rel_full = err_full / ref.abs().max().item()
    res[str(K)] = dict(K=K, Ktail=K - Kt, max_abs_err_vs_torch=err_full,
                       max_rel_err_vs_torch=rel_full,
                       max_abs_err_vs_truncated_ref=err_trunc,
                       allclose_torch_1e3=bool(torch.allclose(c, ref, rtol=1e-3, atol=1e-3)),
                       allclose_trunc_1e3=bool(torch.allclose(c, trunc, rtol=1e-3, atol=1e-3)))
print(json.dumps(res))
