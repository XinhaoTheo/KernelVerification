import torch, sys
sys.path.insert(0, "/root/cases/case_163")
import kernel
dev = "cuda"
def exact_dft(x):
    N = x.shape[1]
    k = torch.arange(N//2+1, device=x.device, dtype=torch.complex128)
    j = torch.arange(N, device=x.device, dtype=torch.complex128)
    M = torch.exp(-2j * torch.pi * torch.outer(k, j).real * 0)  # placeholder
    return None
# simpler correct reference: torch.fft.rfft on float64 (library complex128)
res = {"per_N": {}}
overall_ok = True
for N in (64, 128, 256):
    B = 4
    rows = []
    x = torch.zeros(B, N, dtype=torch.float16, device=dev)
    for b in range(B):
        x[b, b*7 % N] = 1.0
    rows.append(("impulse", x))
    xs = torch.zeros(B, N, dtype=torch.float16, device=dev)
    for b in range(B):
        j = torch.arange(N, device=dev, dtype=torch.float64)
        freq = (b*3+1)
        xs[b] = torch.cos(2*torch.pi*freq*j/N + b).to(torch.float16)
    rows.append(("sinusoid", xs))
    xn = torch.zeros(B, N, dtype=torch.float16, device=dev)
    xn[0] = 1.0; xn[1] = -1.0
    xn[2,:] = (torch.arange(N,device=dev,dtype=torch.float16)%2*2-1).to(torch.float16)
    rows.append(("dc_nyquist_alt", xn))
    rows.append(("random", torch.randn(B, N, device=dev, dtype=torch.float16)))
    bad = 0; worst = 0.0; worst_info = None
    for name, xr in rows:
        out = kernel.run(xr)
        ref = torch.fft.rfft(xr.to(torch.float64), dim=1, norm="backward")
        ex = torch.stack((ref.real, ref.imag), dim=-1)
        err = (out.double() - ex).abs()
        tol = 1e-4 + 1e-6*ex.abs()
        viol = (err > tol)
        if viol.any():
            idx = int((err - tol).argmax())
            bad += int(viol.sum())
            worst_info = {"test": name, "flat_idx": idx, "err": float(err.flatten()[idx]), "tol": float(tol.flatten()[idx])}
        worst = max(worst, (err/tol).max().item())
    res["per_N"][N] = {"worst_err_over_tol": worst, "violations": bad, "worst_case": worst_info}
    overall_ok = overall_ok and bad == 0
res["all_pass"] = overall_ok
print(res)