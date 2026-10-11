import torch, json, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_131/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
bits = np.arange(65536, dtype=np.uint16)
h = torch.from_numpy(bits.view(np.float16).copy())
finite = torch.isfinite(h)
x = h[finite].to(dev)  # all finite FP16 values (includes subnormals, signed zeros, +/-65504)
out = k.run(x)
ref = k.reference(x.cpu()).to(dev)
err = (out - ref).abs()
tol = 1e-6
i = int(err.argmax())
print(json.dumps({
  "metric": "max abs error vs FP64 reference over exhaustive finite FP16 domain",
  "n": x.numel(),
  "max_abs_err": float(err[i]),
  "tolerance": tol,
  "n_violations": int((err > tol).sum()),
  "worst_x": float(x[i].item()),
  "out_at_worst": float(out[i].item()),
  "ref_at_worst": float(ref[i].item()),
  "all_out_finite": bool(torch.isfinite(out).all().item()),
}))