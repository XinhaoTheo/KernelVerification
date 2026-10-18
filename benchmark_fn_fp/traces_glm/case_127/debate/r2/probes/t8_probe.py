import sys, json
sys.path.insert(0, "/root/cases/case_127")
import torch
import kernel as K

# all finite FP16 values except +-inf/nan: keep those 65536 with finite bit patterns
vals = torch.arange(65536, dtype=torch.int32)
bits = vals.to(torch.int16).view(torch.float16).to(torch.float64)
finite = torch.isfinite(bits)
xs = bits[finite]  # includes signed zeros, subnormals, +/-65504
x_gpu = xs.to(torch.float16).to("cuda")
print("n =", x_gpu.numel(), file=sys.stderr)

out = K.run(x_gpu)
out64 = out.to(torch.float64).cpu()

import math
ref = torch.erf(xs)  # FP64 exact erf of the FP64 value of the stored FP16 number

# truncated-coefficient formula evaluated in FP64 (isolates coefficient truncation)
m = xs.abs()
t = 1.0/(1.0+0.3275911*m)
p = 1.06141*t - 1.45315
p = p*t + 1.42141
p = p*t - 0.28450
p = p*t + 0.25483
p = p*t
formula64 = torch.sign(xs)*(1.0 - p*torch.exp(-(m*m)))

err_total = (out64 - ref).abs()
err_coeff = (formula64 - ref).abs()
max_total, i1 = err_total.max(0)
max_coeff, i2 = err_coeff.max(0)
viol = (err_total > 1e-6)
viol_coeff = (err_coeff > 1e-6)
print(json.dumps({
  "n_values": int(x_gpu.numel()),
  "max_abs_err_kernel_vs_exact_erf": float(max_total),
  "argmax_x_kernel": float(xs[i1]),
  "count_violating_1e-6_kernel": int(viol.sum()),
  "max_abs_err_truncated_formula_fp64_vs_exact": float(max_coeff),
  "argmax_x_formula": float(xs[i2]),
  "count_violating_1e-6_truncated_formula": int(viol_coeff.sum()),
  "midrange_violating_kernel_0.3_3": int((viol & (m > 0.3) & (m < 3)).sum()),
  "tolerance": 1e-6,
}))
