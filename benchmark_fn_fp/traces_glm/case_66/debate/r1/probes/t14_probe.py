import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_139/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda" if torch.cuda.is_available() else "cpu"

bits = torch.arange(0, 0x7C00, dtype=torch.int32)  # 0x7C00 is Inf; exclude it
vals = bits.to(torch.uint16).view(torch.float16)
big = vals[vals.float() >= 0.11]
sub_bits = torch.arange(1, 256, dtype=torch.int32).to(torch.uint16).view(torch.float16)
extra = torch.tensor([65504.0, -65504.0, 0.0, -0.0, 1.0, -1.0], dtype=torch.float16)
x = torch.cat([big, -big, sub_bits, -sub_bits, extra]).to(dev)
out = k.run(x)
u = x.to(torch.float64); u2 = u*u
ref = u2 / (torch.sqrt(1.0+u2)+1.0)
err = (out.to(torch.float64) - ref).abs()
tol = 1e-8 + 1e-5*ref.abs()
viol = err > tol
sub_mask = torch.cat([torch.zeros(big.numel()*2, dtype=torch.bool), torch.ones(sub_bits.numel()*2, dtype=torch.bool), torch.zeros(6, dtype=torch.bool)]).to(dev)
print(json.dumps({
  "n_tested": int(x.numel()),
  "num_violations": int(viol.sum()),
  "err_over_tol_worst": float((err/tol).max()),
  "output_finite_all": bool(torch.isfinite(out).all().item()),
  "max_abs_err_large_band": float(err[ref > 1.0].max()) if bool((ref>1.0).any()) else None,
  "extrema_65504_abs_err": float(err[torch.abs(x.float()) == 65504.0].max()),
  "subnormal_max_abs_err": float(err[sub_mask].max()),
}))