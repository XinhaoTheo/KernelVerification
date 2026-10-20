
import torch, json, sys
sys.path.insert(0, "/root/cases/case_100")
import kernel as K

# all finite FP16 values
bits = torch.arange(65536, dtype=torch.int64).to(torch.uint16)
allv = bits.view(torch.float16)
finite = allv[torch.isfinite(allv)]
x = finite.to("cuda")  # includes negatives, -0, +0, subnormals, 65504
ref = torch.asinh(x.to(torch.float64))
out = K.run(x)
err = (out.to(torch.float64) - ref).abs()
mag = x.to(torch.float64).abs()
m1 = mag <= 1.05
e1 = err[m1]
i1 = e1.argmax()
w1 = x[m1][i1].item()
res = {
  "metric": "max_abs_error vs FP64 asinh, exhaustive over all finite FP16 values, polynomial branch subset",
  "tolerance": 1e-6,
  "n_total_finite": int(mag.numel()),
  "n_poly_branch": int(m1.sum()),
  "poly_max_abs_err": float(e1.max()),
  "poly_worst_input": w1,
  "poly_worst_ref": float(ref[m1][i1]),
  "poly_worst_out": float(out[m1][i1]),
  "poly_count_over_tol": int((e1 > 1e-6).sum()),
  "poly_err_near_1.05": float(err[(mag > 1.0) & (mag <= 1.05)].max()),
}
print(json.dumps(res))
