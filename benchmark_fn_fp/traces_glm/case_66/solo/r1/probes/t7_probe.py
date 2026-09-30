import json, itertools, numpy as np, torch, sys
sys.path.insert(0, "/root/evidence_cases/case_e05")
import kernel
inputs = kernel.make_inputs()
before = [t.clone() for t in inputs]
y = kernel.run(*inputs)
v = y.detach().cpu().numpy()
x = inputs[1].detach().cpu().numpy()
inputs_unmod = all(torch.equal(a,b) for a,b in zip(inputs,before))
allowed = np.logical_or(v==0, v==2*x[None,:]).all()
B = (v != 0).astype(np.int64)
worst = 0.0; worst_case = None
for quad in itertools.combinations(range(8),4):
    codes = (B[:,quad[0]]*8 + B[:,quad[1]]*4 + B[:,quad[2]]*2 + B[:,quad[3]])
    counts = np.bincount(codes, minlength=16)
    err = float(np.max(np.abs(counts/1024 - 1/16)))
    if err > worst:
        worst = err; worst_case = quad
    if err > 1/1024 + 1e-12:
        pass
print(json.dumps({
  "metric":"4th-order joint pattern histogram error, all 70 quadruples x 16 patterns, metric = |count/1024 - 1/16|",
  "max_4th_order_error": worst,
  "worst_quadruple": worst_case,
  "tolerance": 1/1024,
  "shape_dtype_finite": bool(v.shape==(1024,8) and v.dtype==np.float32 and np.isfinite(v).all()),
  "values_allowed": bool(allowed),
  "inputs_unmodified": bool(inputs_unmod),
  "passes_contract": bool(v.shape==(1024,8) and allowed and worst <= 1/1024 + 1e-12)
}))