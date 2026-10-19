
import torch, json, sys
sys.path.insert(0, "/root/cases/case_147")
import kernel

# Upper branch: FP16 values just above 2.414213562373095 and up through moderate |x|
v = torch.arange(2.0, 8.0, 0.0005, dtype=torch.float64)
xf = v.to(torch.float16).unique().to(torch.float64)
mag = xf.abs()
up = mag > 2.414213562373095
xf = xf[up]
x = torch.cat([(-xf).to(torch.float16), xf.to(torch.float16)]).to("cuda")
out = kernel.run(x)
ref = torch.atan(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
worst = torch.argmax(err)
near_thr = x.to(torch.float64).abs() < 2.5
print(json.dumps({
    "n": int(x.numel()),
    "tol": 1e-6,
    "max_abs_err_upper": float(err.max().item()),
    "n_over_tol": int((err > 1e-6).sum().item()),
    "max_abs_err_near_thr": float(err[near_thr].max().item()),
    "n_over_tol_near_thr": int((err[near_thr] > 1e-6).sum().item()),
    "worst_x": float(x[worst].item()),
    "worst_err": float(err[worst].item()),
}))
