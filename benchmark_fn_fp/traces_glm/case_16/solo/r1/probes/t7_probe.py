import torch, json, sys
sys.path.insert(0, '/root/cases/case_16')
from kernel import matmul

torch.manual_seed(0)
res = {}
for K in [32, 33, 70, 100]:
    a = torch.randn(64, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, 64, device='cuda', dtype=torch.float32)
    c = matmul(a, b)
    ref = a @ b
    err = (c - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    res[K] = {'max_abs_err': err, 'rel_err': rel, 'K_mod_32': K % 32}
print(json.dumps(res))