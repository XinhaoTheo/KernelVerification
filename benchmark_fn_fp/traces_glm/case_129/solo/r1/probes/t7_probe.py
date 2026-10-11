import torch, math, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_129/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

bits = torch.arange(0, 65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
f = bits[torch.isfinite(bits)]
f64 = f.to(torch.float64)
ref = 0.5 * f64 * torch.erfc(-f64 / math.sqrt(2.0))

out = k.run(f.cuda()).cpu().to(torch.float64)
err = (out - ref).abs()
i = int(err.argmax())
print(json.dumps({
  "metric": "max abs error over all finite FP16 inputs vs FP64 erfc reference",
  "n_finite": f.numel(),
  "max_err": float(err[i]),
  "worst_x": float(f[i]),
  "out": float(out[i]),
  "ref": float(ref[i]),
  "num_violations_gt_1e-6": int((err > 1e-6).sum()),
  "num_violations_gt_1e-5": int((err > 1e-5).sum()),
}))
