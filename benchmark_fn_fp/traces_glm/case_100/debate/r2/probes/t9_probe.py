import torch, json, sys
sys.path.insert(0, "/root/cases/case_100")
import kernel

# All stored FP16 values with magnitude <= 1.05 (the polynomial branch region)
bits = torch.arange(0, 65536, dtype=torch.int32)
all_fp16 = bits.to(torch.uint16).view(torch.float16)
finite = all_fp16[torch.isfinite(all_fp16)]
mag = finite.float().abs()
sel = finite[(mag <= 1.05)]
x = sel.cuda().contiguous()
out = kernel.run(x)
ref = torch.asinh(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
tol = 1e-6
bad = err > tol
print(json.dumps({
  "metric": "max abs error vs FP64 asinh over all stored FP16 magnitudes <= 1.05",
  "reason": "Contract requires abs(actual-asinh(x)) <= 1e-6 for every finite FP16 value; polynomial branch covers |x|<=1.05",
  "n_tested": int(x.numel()),
  "max_abs_err": float(err.max()),
  "tolerance": tol,
  "n_violations": int(bad.sum()),
  "worst_x": float(sel[err.argmax()]),
  "worst_x_err": float(err.max()),
}))
