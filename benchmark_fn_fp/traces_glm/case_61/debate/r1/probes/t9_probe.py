import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_134/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# all finite FP16 bit patterns (exclude NaN/Inf: exponent 0x1F and mantissa !=0)
bits = torch.arange(65536, dtype=torch.int32)
u16 = (bits & 0xFFFF).to(torch.uint16).view(torch.float16)
finite = torch.isfinite(u16)
x = u16[finite].cuda()  # 30720? actually finite count = 65536 - 2(inf sign) - 1024*2(nan) -> compute
n = x.numel()
out = k.run(x)
ref = k.reference(x)
err = (out.double() - ref).abs()
i = int(err.argmax())
res = {
  "n_finite": n, "max_abs_err": float(err[i]), "tolerance": 1e-6,
  "worst_x": float(x[i]), "worst_out": float(out[i]), "worst_ref": float(ref[i]),
  "num_exceed": int((err > 1e-6).sum()),
  "num_exceed_1e_7": int((err > 1e-7).sum()),
  "max_err_at_65504": float(err[torch.searchsorted(x.cpu().sort().values, torch.tensor(65504., dtype=torch.float16)).item()])
}
print(json.dumps(res))