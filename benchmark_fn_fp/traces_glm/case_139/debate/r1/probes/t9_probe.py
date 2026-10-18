import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_139/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# all positive FP16 values in mid-band magnitude [1e-3, 0.11], plus negatives
xs = []
v = torch.finfo(torch.float16).tiny  # 6.1e-5
# iterate all positive fp16 values via view of uint16 bits
bits = torch.arange(0, 0x7C01, dtype=torch.int32)
vals = bits.to(torch.uint16).view(torch.float16)  # all positive finite fp16
mask = (vals.float() >= 1e-3) & (vals.float() <= 0.11)
xs = vals[mask]
x = torch.cat([xs, -xs]).to(dev)
out = k.run(x)
u = x.to(torch.float64); u2 = u*u
ref = u2 / (torch.sqrt(1.0+u2)+1.0)
err = (out.to(torch.float64) - ref).abs()
tol = 1e-8 + 1e-5*ref.abs()
viol = err > tol
i = int(torch.argmax(err - tol))
print(json.dumps({
  "n_tested": int(x.numel()),
  "num_violations": int(viol.sum()),
  "worst_x": float(x.flatten()[i]),
  "worst_abs_err": float(err.flatten()[i]),
  "tolerance_at_worst": float(tol.flatten()[i]),
  "err_over_tol_worst": float((err/tol).max()),
  "rho_at_worst": float(ref.flatten()[i]),
}))
