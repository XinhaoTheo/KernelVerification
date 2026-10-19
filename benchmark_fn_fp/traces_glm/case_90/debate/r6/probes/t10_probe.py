import torch, sys
sys.path.insert(0, "/root/cases/case_163")
import kernel
dev = "cuda"
def exact_dft(x):
    # x: [B,N] float64 -> [B,N//2+1,2] high precision
    N = x.shape[1]
    j = torch.arange(N, dtype=torch.float64, device=x.device)
    k = torch.arange(N//2+1, dtype=torch.float64, device=x.device)
    ang = -2j * torch.pi * torch.outer(k, j) / N
    M = torch.exp(ang)  # [K,N]
    X = x @ M.T.conj().conj().T  # x [B,N] @ exp matrix [N,K]
    return torch.stack((X.real, X.imag), dim=-1)

res = {"per_N": {}}
overall_ok = True
for N in (64, 128, 256):
    rows = []
    B = 4
    x = torch.zeros(B, N, dtype=torch.float16, device=dev)
    # impulse at different positions
    for b in range(B):
        x[b, b*7 % N] = 1.0
    rows.append(x)
    # adversarial spectra: single-bin complex sinusoid rows
    xs = torch.zeros(B, N, dtype=torch.float16, device=dev)
    for b in range(B):
        j = torch.arange(N, device=dev, dtype=torch.float64)
        freq = (b*3+1)
        xs[b] = torch.cos(2*torch.pi*freq*j/N + b).to(torch.float16)
    rows.append(xs)
    # Nyquist / DC / alternating
    xn = torch.zeros(B, N, dtype=torch.float16, device=dev)
    xn[0] = 1.0; xn[1] = -1.0; xn[2,:] = (torch.arange(N,device=dev,dtype=torch.float16)%2*2-1)
    rows.append(xn)
    # random
    rows.append(torch.randn(B, N, device=dev, dtype=torch.float16))
    worst = 0.0; worst_rel_ok = True; bad = 0; worst_info = None
    for ti, x in enumerate(rows):
        out = kernel.run(x)
        ex = exact_dft(x.to(torch.float64))
        err = (out.double() - ex).abs()
        tol = 1e-4 + 1e-6*ex.abs()
        viol = (err > tol)
        if viol.any():
            bad += int(viol.sum())
            idx = (err - tol).argmax()
            worst_info = {"test": ti, "idx": idx.item(), "err": err.flatten()[idx].item(), "tol": tol.flatten()[idx].item()}
        r = (err / (tol)).max().item()
        worst = max(worst, r)
    res["per_N"][N] = {"worst_err_over_tol": worst, "violations": bad, "worst_case": worst_info}
    overall_ok = overall_ok and bad == 0
res["all_pass"] = overall_ok
print(res)