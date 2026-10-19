
import torch, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel

# All finite FP16 magnitudes in middle interval (0.66 < |x| <= 2.414213562373095)
# plus lower-interval values near the 0.66 seam (0.6..2.5 covers both sides).
vals = []
v = torch.arange(0.5, 2.5, 0.00005, dtype=torch.float64)
xf = v.to(torch.float16).unique().to(torch.float64)
mask = (xf > 0.5) & (xf < 2.5)
xf = xf[mask]
x = torch.cat([(-xf).to(torch.float16), xf.to(torch.float16)]).to("cuda")
out = kernel.run(x)
ref = torch.atan(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
# focus: middle interval elements
mag = x.to(torch.float64).abs()
mid = (mag > 0.66) & (mag <= 2.414213562373095)
seam = (mag > 0.63) & (mag < 0.70)
miderr = err[mid]
worst = torch.argmax(err)
print(json.dumps({
    "n": int(x.numel()),
    "n_mid": int(mid.sum().item()),
    "tol": 1e-6,
    "max_abs_err_mid": float(miderr.max().item()) if miderr.numel() else None,
    "n_over_tol_mid": int((miderr > 1e-6).sum().item()) if miderr.numel() else None,
    "max_abs_err_seam": float(err[seam].max().item()) if seam.sum() else None,
    "max_abs_err_all": float(err.max().item()),
    "worst_x": float(x[worst].item()),
    "worst_err": float(err[worst].item()),
}))
