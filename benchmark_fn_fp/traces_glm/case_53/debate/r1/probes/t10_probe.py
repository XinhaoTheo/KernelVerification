import torch, json
import sys
sys.path.insert(0, "/root/cases/case_126")
import kernel as K

bits = torch.arange(0, 65536, dtype=torch.int32)
x_all = bits.to(torch.uint16).view(torch.float16)
finite = torch.isfinite(x_all)
xf = x_all[finite]
# small magnitude: subnormals, zeros, |x| < 1e-2
mask = (xf == 0) | (xf.abs() < 1e-2)
x = xf[mask].cuda()
out = K.run(x)
ref = torch.erf(x.to(torch.float64)).to(torch.float32)
err = (out.double() - ref.double()).abs()
print(json.dumps({
  "metric": "max_abs_error_small_subnormal_inputs",
  "n": int(x.numel()),
  "max_abs_error": err.max().item(),
  "tolerance": 1e-6,
  "violations": int((err > 1e-6).sum().item()),
  "max_rel_error_small": (err / ref.abs().clamp_min(1e-300)).max().item() if (ref != 0).any() else None,
  "argmax_x": x[err.argmax()].item() if x.numel() else None,
  "kernel_out_at_argmax": out[err.argmax()].item() if x.numel() else None,
}))