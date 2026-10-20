
import torch, json, sys
sys.path.insert(0, "/root/cases/case_100")
import kernel as K

bits = torch.arange(65536, dtype=torch.int64).to(torch.uint16)
allv = bits.view(torch.float16)
finite = allv[torch.isfinite(allv)]
x = finite.to("cuda")
ref = torch.asinh(x.to(torch.float64))
out = K.run(x)
err = (out.to(torch.float64) - ref).abs()
mag = x.to(torch.float64).abs()
m2 = mag > 1.05
e2 = err[m2]
i2 = e2.argmax()
res = {
  "metric": "max_abs_error vs FP64 asinh, exhaustive over all finite FP16 values, outer log branch subset",
  "tolerance": 1e-6,
  "n_outer_branch": int(m2.sum()),
  "outer_max_abs_err": float(e2.max()),
  "outer_worst_input": x[m2][i2].item(),
  "outer_worst_ref": float(ref[m2][i2]),
  "outer_worst_out": float(out[m2][i2]),
  "outer_count_over_tol": int((e2 > 1e-6).sum()),
  "outer_err_just_above_seam_(1.05,2)": float(err[(mag > 1.05) & (mag <= 2.0)].max()),
  "outer_err_large_(>1000)": float(err[mag > 1000].max()),
  "err_at_65504": float(err[mag == 65504].max()),
}
print(json.dumps(res))
