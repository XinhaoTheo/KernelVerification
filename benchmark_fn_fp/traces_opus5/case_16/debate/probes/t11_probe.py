
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_16/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(2)
res = {}
for (M, N, K) in [(70, 100, 64), (65, 65, 32), (100, 70, 128)]:
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    try:
        c = m.matmul(a, b)
        torch.cuda.synchronize()
        ref = (a.double() @ b.double()).float()
        err = (c - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        # poison test: same logical matrices inside a larger buffer with garbage past the end
        bigA = torch.full((M + 8, K), 1e6, device='cuda', dtype=torch.float32); bigA[:M] = a
        bigB = torch.full((K, N + 8), 1e6, device='cuda', dtype=torch.float32); bigB[:, :N] = b
        c2 = m.matmul(bigA[:M].contiguous() if False else a, b)
        # true poison: view into larger allocation so OOB reads hit 1e6
        vA = bigA.as_strided((M, K), (K, 1))
        vB = bigB.as_strided((K, N), (N + 8, 1))
        c3 = m.matmul(vA, vB)
        torch.cuda.synchronize()
        err_poison = (c3 - ref).abs().max().item()
        res[f"{M}x{N}x{K}"] = dict(M=M, N=N, K=K, max_abs_err=err, max_rel_err=rel,
                                   allclose_1e3=bool(torch.allclose(c, ref, rtol=1e-3, atol=1e-3)),
                                   max_abs_err_poisoned_alloc=err_poison,
                                   poison_allclose_1e3=bool(torch.allclose(c3, ref, rtol=1e-3, atol=1e-3)),
                                   fault=False)
    except Exception as e:
        res[f"{M}x{N}x{K}"] = dict(M=M, N=N, K=K, fault=True, error=repr(e)[:300])
print(json.dumps(res))
