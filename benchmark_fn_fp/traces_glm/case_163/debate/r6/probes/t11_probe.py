import torch, sys
sys.path.insert(0, "/root/cases/case_163")
import kernel
dev = "cuda"
def exact_dft(x):
    N = x.shape[1]
    j = torch.arange(N, dtype=torch.float64, device=x.device)
    k = torch.arange(N//2+1, dtype=torch.float64, device=x.device)
    M = torch.exp(-2j * torch.pi * torch.outer(k, j) / N)
    X = x @ M.T
    return torch.stack((X.real, X.imag), dim=-1)

# verify cosine table literals at double precision
import math
tab_ok = True; tab_maxdiff = 0.0
from kernel import _COSINE
for i, v in enumerate(_COSINE):
    d = abs(v - math.cos(2*math.pi*i/256))
    tab_maxdiff = max(tab_maxdiff, d)
    tab_ok = tab_ok and d < 1e-17
res = {"table_maxdiff": tab_maxdiff, "table_ok": tab_ok, "cases": {}}
worst = 0.0; total_bad = 0
for N in (64, 128, 256):
    B = 6
    x = torch.zeros(B, N, dtype=torch.float16, device=dev)
    big = torch.finfo(torch.float16).max
    x[0,:] = big
    x[1,:] = -big
    x[2,:] = big if N % 2 == 0 else 0
    x[2,:] = torch.where(torch.arange(N,device=dev)%2==0, torch.tensor(big,device=dev), torch.tensor(-big,device=dev))
    x[3,:] = torch.linspace(big, 1.0, N, device=dev).to(torch.float16)
    x[4] = (torch.randn(N, device=dev) * 20000).to(torch.float16)
    x[5,:] = 6e-8  # subnormal-ish small values (fp16 subnormal min ~6e-8)
    x[5,:] = 5.96e-8
    out = kernel.run(x)
    ex = exact_dft(x.to(torch.float64))
    err = (out.double() - ex).abs()
    tol = 1e-4 + 1e-6*ex.abs()
    viol = int((err > tol).sum())
    ratio = (err/tol).max().item()
    fin = bool(torch.isfinite(out).all())
    res["cases"][N] = {"violations": viol, "worst_err_over_tol": ratio, "finite": fin, "max_exact_mag": ex.abs().max().item()}
    total_bad += viol
    worst = max(worst, ratio)
res["all_pass"] = total_bad == 0
res["worst_ratio"] = worst
print(res)