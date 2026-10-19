import torch, sys, json
sys.path.insert(0, "/root/cases/case_134")
import kernel

bits = torch.arange(65536, dtype=torch.int32)
allvals = bits.to(torch.uint16).view(torch.float16)
finite = allvals[torch.isfinite(allvals)]
# NaN/Inf excluded by contract
x = finite.to("cuda")
out = kernel.run(x)
tgt = x.to(torch.float64) / torch.sqrt(1.0 + x.to(torch.float64) * x.to(torch.float64))
err = (out.to(torch.float64) - tgt).abs()
mx = err.max().item()
worst = int(err.argmax())
print(json.dumps({
  "metric": "max absolute error over all finite FP16 values",
  "num_finite": int(finite.numel()),
  "max_abs_err": mx,
  "worst_input_fp16": float(finite[worst]),
  "worst_target": float(tgt[worst]),
  "worst_actual": float(out[worst]),
  "count_violations_1e6": int((err > 1e-6).sum()),
  "tolerance": 1e-6,
  "tolerance_met": bool(mx <= 1e-6),
}))
