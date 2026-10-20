
import sys, torch, json
sys.path.insert(0, "/root/cases/case_166")
import kernel

bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
allf16 = bits.view(torch.float16).to("cuda")
legal = allf16[(torch.isfinite(allf16.cpu())) & (allf16.cpu() > -1)] if False else None
# build on CPU then move
bits_cpu = torch.arange(65536, dtype=torch.int32)
allf = bits_cpu.to(torch.int32).to(torch.uint16).view(torch.float16)
mask = torch.isfinite(allf) & (allf > torch.tensor(-1.0, dtype=torch.float16))
x = allf[mask]
# keep only large values for this claim's tensor (N<=131072 anyway)
x = x[x >= 1.0].to("cuda")
print("N large-x:", x.numel())
out = kernel.run(x)
ref = torch.log1p(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
maxerr, idx = err.max(0)
json.dump({
  "claim": "c1 large-x accumulated rounding",
  "n": int(x.numel()),
  "max_abs_err": float(maxerr),
  "worst_x": float(x[idx]),
  "worst_output": float(out[idx]),
  "tolerance": 1e-6,
  "exceeds_tol": bool(maxerr > 1e-6),
  "n_violations": int((err > 1e-6).sum())
}, sys.stdout)
