
import torch, math, json, importlib.util, os
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_129/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# all finite FP16 values
bits = torch.arange(0, 65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
f = bits[torch.isfinite(bits)]
print("finite fp16 count:", f.numel())

f64 = f.to(torch.float64)
ref = 0.5 * f64 * torch.erfc(-f64 / math.sqrt(2.0))

out = k.run(f.cuda())
err = (out.to(torch.float64) - ref).abs()
i = int(err.argmax())
print(json.dumps({
  "metric": "max abs error over all finite FP16 inputs",
  "max_err": float(err[i]),
  "argmax_x": float(f[i]),
  "num_violations_1e-6": int((err > 1e-6).sum()),
  "num_violations_1e_5": int((err > 1e-5).sum()),
  "num_violations_1e_7": int((err > 1e-7).sum()),
  "worst_x": float(f[i]),
  "out": float(out[i]),
  "ref": float(ref[i]),
}))
