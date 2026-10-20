
import sys, torch, json
sys.path.insert(0, "/root/cases/case_166")
import kernel

bits = torch.arange(65536, dtype=torch.int32)
allf = bits.to(torch.int32).to(torch.uint16).view(torch.float16)
mask = torch.isfinite(allf) & (allf > torch.tensor(-1.0, dtype=torch.float16))
# regime of interest: fl(1+x) in (1.6, 2.05), i.e. x in (0.55, 1.05) roughly,
# plus u near 2 boundary: x near 1.0
x = allf[mask]
sel = x[(x > 0.5) & (x < 1.5)]
xg = sel.to("cuda")
out = kernel.run(xg)
ref = torch.log1p(xg.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
maxerr, idx = err.max(0)
json.dump({
  "claim": "c2 polynomial/coefficient error near split boundary and near u=2",
  "n": int(xg.numel()),
  "max_abs_err": float(maxerr),
  "worst_x": float(xg[idx]),
  "worst_output": float(out[idx]),
  "tolerance": 1e-6,
  "exceeds_tol": bool(maxerr > 1e-6),
  "n_violations": int((err > 1e-6).sum())
}, sys.stdout)
