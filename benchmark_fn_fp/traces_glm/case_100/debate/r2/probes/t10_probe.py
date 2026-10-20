import torch, json, sys
sys.path.insert(0, "/root/cases/case_100")
import kernel

# Large-magnitude stored FP16 values (outer log branch)
bits = torch.arange(0, 65536, dtype=torch.int32)
all_fp16 = bits.to(torch.uint16).view(torch.float16)
finite = all_fp16[torch.isfinite(all_fp16)]
mag = finite.float().abs()
sel = finite[(mag >= 100.0)]  # outer branch, incl. +/-65504
x = sel.cuda().contiguous()
out = kernel.run(x)
ref = torch.asinh(x.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
tol = 1e-6
bad = err > tol
print(json.dumps({
  "metric": "max abs error vs FP64 asinh over all stored FP16 |x|>=100 (incl. +/-65504)",
  "reason": "Contract includes largest finite FP16 values with fixed absolute 1e-6 tolerance; outer branch is log(|x|+sqrt(1+x^2)) in FP32",
  "n_tested": int(x.numel()),
  "max_abs_err": float(err.max()),
  "tolerance": tol,
  "n_violations": int(bad.sum()),
  "worst_x": float(sel[err.argmax()]),
  "asinh_65504_err": float(err[mag[finite>=0] if False else err.argmax()]),
}))
