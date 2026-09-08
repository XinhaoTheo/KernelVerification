
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_16/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(1)
res = {}
for K in [1, 8, 31]:
    M = N = 64
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    c = m.matmul(a, b)
    ref = a @ b
    res[str(K)] = dict(K=K, kernel_all_zero=bool((c == 0).all().item()),
                       kernel_abs_max=c.abs().max().item(),
                       ref_abs_max=ref.abs().max().item(),
                       max_abs_err=(c - ref).abs().max().item(),
                       nonzero_frac=float((c != 0).float().mean().item()))
print(json.dumps(res))
