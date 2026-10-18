import sys, json
sys.path.insert(0, "/root/cases/case_127")
import torch
import kernel as K

vals = torch.arange(65536, dtype=torch.int32)
bits = vals.to(torch.int16).view(torch.float16).to(torch.float64)
xs = bits[torch.isfinite(bits)]
x_gpu = xs.to(torch.float16).to("cuda")

out = K.run(x_gpu)
out64 = out.to(torch.float64).cpu()

# same formula, same truncated coefficients, evaluated in FP64:
# difference kernel vs this isolates the FP32 evaluation chain (div_rn, tl.exp, Horner, fusion off)
m = xs.abs()
t = 1.0/(1.0+0.3275911*m)
p = 1.06141*t - 1.45315
p = p*t + 1.42141
p = p*t - 0.28450
p = p*t + 0.25483
p = p*t
formula64 = torch.sign(xs)*(1.0 - p*torch.exp(-(m*m)))

err_eval = (out64 - formula64).abs()
err_total = (out64 - torch.erf(xs)).abs()
max_eval, ie = err_eval.max(0)
print(json.dumps({
  "n_values": int(x_gpu.numel()),
  "max_abs_err_kernel_vs_fp64_formula": float(max_eval),
  "argmax_x": float(xs[ie]),
  "count_eval_err_gt_1e-6": int((err_eval > 1e-6).sum()),
  "count_eval_err_gt_1e-7": int((err_eval > 1e-7).sum()),
  "count_eval_err_gt_5e-7": int((err_eval > 5e-7).sum()),
  "max_abs_err_kernel_vs_exact_erf": float(err_total.max()),
  "tolerance": 1e-6,
}))
