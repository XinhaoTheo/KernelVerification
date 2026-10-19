import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_139/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# extrema and subnormals, plus large-x band sweep (|x|>=0.11 up to 65504)
bits = torch.arange(0, 0x7C01, dtype=torch.int32)
vals = bits.to(torch.uint16).view(torch.float16)
big = vals[vals.float() >= 0.11]
small = vals[vals.float() < torch.finfo(torch.float16).tiny*0]  # empty
# subnormals: bits 1..255 (fp16)
sub_bits = torch.arange(1, 256, dtype=torch.int32).to(torch.uint16).view(torch.float16)
extra = torch.tensor([65504.0, -65504.0, 0.0, -0.0, 1.0, -1.0], dtype=torch.float16)
x = torch.cat([big, -big, sub_bits, -sub_bits, extra]).to(dev)
out = k.run(x)
u = x.to(torch.float64); u2 = u*u
ref = u2 / (torch.sqrt(1.0+u2)+1.0)
err = (out.to(torch.float64) - ref).abs()
tol = 1e-8 + 1e-5*ref.abs()
viol = err > tol
print(json.dumps({
  "n_tested": int(x.numel()),
  "num_violations": int(viol.sum()),
  "err_over_tol_worst": float((err/tol).max()),
  "output_finite_all": bool(torch.isfinite(out).all().item()),
  "max_abs_err_large_band": float(err[ref > 1.0].max()) if bool((ref>1.0).any()) else None,
  "subnormal_count": int(sub_bits.numel()*2),
  "subnormal_max_abs_err": float(err[:sub_bits.numel()*2].max()),
}))
