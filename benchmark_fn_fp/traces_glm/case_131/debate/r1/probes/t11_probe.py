import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_131/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"
torch.manual_seed(0)
base = torch.randn(5000, dtype=torch.float32).to(torch.float16).to(dev)
n = 4093; off = 511
x = base[off:off+n]  # contiguous view, nonzero storage offset
assert x.is_contiguous() and x.storage_offset() == off
out = k.run(x)
ref = k.reference(x.cpu()).to(dev)
err = (out - ref).abs()
i = int(err.argmax())
# also confirm output matches SiLU of the viewed elements (not base[0:n])
ref_base0 = k.reference(base[:n].cpu()).to(dev)
print(json.dumps({
  "metric": "max abs error of run(view) vs reference(view)",
  "n": n, "offset": off,
  "max_abs_err": float(err[i]),
  "tolerance": 1e-6,
  "n_violations": int((err > 1e-6).sum()),
  "mismatch_vs_base0_prefix": float((out - ref_base0).abs().max()),
  "input_preserved": bool(torch.equal(x, base[off:off+n].cpu().to(dev)) if False else torch.equal(x.cpu(), base[off:off+n].cpu())),
}))