import torch, json
import sys
sys.path.insert(0, "/root/cases/case_126")
import kernel as K

# all finite FP16 bit patterns
bits = torch.arange(0, 65536, dtype=torch.int32)
x_all = bits.to(torch.uint16).view(torch.float16)
finite = torch.isfinite(x_all)
x = x_all[finite].cuda()  # 65504 finite values
out = K.run(x)
ref = torch.erf(x.to(torch.float64)).to(torch.float32)
err = (out.double() - ref.double()).abs()
maxerr = err.max().item()
imax = err.argmax().item()
viol = int((err > 1e-6).sum().item())
xs = x[imax].item()
print(json.dumps({
  "metric": "max_abs_error_vs_fp64_erf_all_finite_fp16",
  "n_finite": x.numel(),
  "max_abs_error": maxerr,
  "tolerance": 1e-6,
  "violations": viol,
  "argmax_x": xs,
  "kernel_out_at_argmax": out[imax].item(),
  "ref_at_argmax": ref[imax].item(),
}))