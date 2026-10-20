import torch, json, sys
sys.path.insert(0, "/root/cases/case_100")
import kernel

# Stored FP16 values near the branch threshold |x| = 1.05 (both sides)
bits = torch.arange(0, 65536, dtype=torch.int32)
all_fp16 = bits.to(torch.uint16).view(torch.float16)
finite = all_fp16[torch.isfinite(all_fp16)]
mag = finite.float().abs()
sel = finite[(mag >= 0.95) & (mag <= 1.15)]
x = sel.cuda().contiguous()
out = kernel.run(x)
ref = torch.asinh(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
tol = 1e-6
bad = err > tol
below = mag[(finite>=0) & (mag>=0.95) & (mag<=1.05)]  # magnitudes below/at threshold
above = mag[(finite>=0) & (mag>1.05) & (mag<=1.15)]
print(json.dumps({
  "metric": "max abs error vs FP64 asinh over stored FP16 0.95<=|x|<=1.15 straddling branch threshold 1.05",
  "reason": "Branch-boundary consistency: both polynomial (<=1.05) and log (>1.05) results must stay within 1e-6 for values straddling the threshold",
  "n_tested": int(x.numel()),
  "n_below_threshold": int(below.numel()),
  "n_above_threshold": int(above.numel()),
  "max_abs_err": float(err.max()),
  "tolerance": tol,
  "n_violations": int(bad.sum()),
  "worst_x": float(sel[err.argmax()]),
}))
