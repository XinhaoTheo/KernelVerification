import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_131/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
# FP16 values in |x| in [1,4], dense
vals = torch.tensor([t/1024.0 for t in range(1024, 4097)], dtype=torch.float32).to(torch.float16)
x = torch.cat([vals, -vals]).to(dev)
out = k.run(x)
ref = k.reference(x.cpu().cuda() if False else x).cpu()
ref = k.reference(x.cpu()).to(dev)
err = (out - ref).abs()
tol = 1e-6
bad = (err > tol)
i = int(err.argmax())
print(json.dumps({
  "metric": "max abs error vs FP64 reference",
  "n": x.numel(),
  "max_abs_err": float(err[i]),
  "tolerance": tol,
  "n_violations": int(bad.sum()),
  "worst_x": float(x[i].item()),
  "worst_x_abs": float(abs(x[i].item())),
  "out_at_worst": float(out[i].item()),
  "ref_at_worst": float(ref[i].item()),
  "x_band": "[1,4] dense, both signs",
}))