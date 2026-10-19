import torch, sys, math
sys.path.insert(0, "/root/cases/case_163")
import kernel
from kernel import _COSINE
dev = "cuda"
tab_maxdiff = max(abs(v - math.cos(2*math.pi*i/256)) for i, v in enumerate(_COSINE))
res = {"table_maxdiff": tab_maxdiff, "cases": {}}
total_bad = 0; worst = 0.0
for N in (64, 128, 256):
    B = 6
    big = torch.finfo(torch.float16).max
    x = torch.zeros(B, N, dtype=torch.float16, device=dev)
    x[0,:] = big
    x[1,:] = -big
    x[2,:] = torch.where(torch.arange(N,device=dev)%2==0, torch.tensor(big,device=dev), torch.tensor(-big,device=dev)).to(torch.float16)
    x[3,:] = torch.linspace(big, 1.0, N, device=dev).to(torch.float16)
    x[4] = (torch.randn(N, device=dev) * 20000).to(torch.float16)
    x[5,:] = 5.96e-8
    out = kernel.run(x)
    ref = torch.fft.rfft(x.to(torch.float64), dim=1, norm="backward")
    ex = torch.stack((ref.real, ref.imag), dim=-1)
    err = (out.double() - ex).abs()
    tol = 1e-4 + 1e-6*ex.abs()
    viol = int((err > tol).sum())
    ratio = (err/tol).max().item()
    fin = bool(torch.isfinite(out).all())
    res["cases"][N] = {"violations": viol, "worst_err_over_tol": ratio, "finite": fin, "max_exact_mag": float(ex.abs().max())}
    total_bad += viol
    worst = max(worst, ratio)
res["all_pass"] = total_bad == 0
res["worst_ratio"] = worst
print(res)