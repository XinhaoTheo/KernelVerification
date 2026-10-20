
import sys, torch, json
sys.path.insert(0, "/root/cases/case_166")
import kernel

bits = torch.arange(65536, dtype=torch.int32)
allf = bits.to(torch.int32).to(torch.uint16).view(torch.float16)
mask = torch.isfinite(allf) & (allf > torch.tensor(-1.0, dtype=torch.float16))
x = allf[mask]
# regime: x in (-1, -0.5] plus the value immediately above -1; also include (-1,0) all
sel = x[x < 0.0]
xg = sel.to("cuda")
out = kernel.run(xg)
ref = torch.log1p(xg.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
maxerr, idx = err.max(0)
# also check the immediate value above -1
next_up = x[x > -1.0].min()
json.dump({
  "claim": "c3 u=fl(1+x) rounding near x=-1",
  "n": int(xg.numel()),
  "max_abs_err": float(maxerr),
  "worst_x": float(xg[idx]),
  "worst_output": float(out[idx]),
  "fp16_next_above_-1": float(next_up),
  "err_at_next_above_-1": float(err[torch.nonzero(xg == next_up, as_tuple=True)[0][0]]) if (xg == next_up).any() else None,
  "tolerance": 1e-6,
  "exceeds_tol": bool(maxerr > 1e-6),
  "n_violations": int((err > 1e-6).sum())
}, sys.stdout)
