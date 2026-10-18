import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_139/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# all positive fp16 with |x| < 0.011 (u=x*x < half-ulp(1.0)=5.96e-8)
bits = torch.arange(0, 0x7C01, dtype=torch.int32)
vals = bits.to(torch.uint16).view(torch.float16)
mask = vals.float() < 0.011
xs = vals[mask]
x = torch.cat([xs, -xs]).to(dev)
out = k.run(x)
u = x.to(torch.float64); u2 = u*u
ref = u2 / (torch.sqrt(1.0+u2)+1.0)
err = (out.to(torch.float64) - ref).abs()
tol = 1e-8 + 1e-5*ref.abs()
zero_out = (out == 0.0)
nonzero_ref = ref != 0
snapped = (zero_out & nonzero_ref)
print(json.dumps({
  "n_tested": int(x.numel()),
  "num_snap_to_zero_nonzero_rho": int(snapped.sum()),
  "num_violations": int((err>tol).sum()),
  "worst_abs_err": float(err.max()),
  "tol_at_worst": float(tol.max()),
  "err_over_tol_worst": float((err/tol).max()),
  "all_within_1e-8_abs": bool((err <= 1e-8).all().item()),
}))
