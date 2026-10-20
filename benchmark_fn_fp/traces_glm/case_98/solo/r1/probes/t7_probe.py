import torch, numpy as np, json, sys
sys.path.insert(0, "/root/cases/case_98")
import kernel

bits = np.arange(0, 65536, dtype=np.uint16)
v = torch.from_numpy(bits.view(np.float16).copy()).to(torch.float32)
mask = torch.isfinite(v) & (v.abs() <= 10.0)
vals = torch.from_numpy(bits[mask.numpy()].copy()).to(torch.float16).to("cuda")
n = vals.numel()
print("num legal fp16 values:", n)

out = kernel.run(vals)
ref = torch.expm1(vals.to(torch.float64))
err = (out.to(torch.float64) - ref).abs()
tol = 1e-7 + 1e-6 * ref.abs()
bad = err > tol
print("num violations:", int(bad.sum()))
if int(bad.sum()) > 0:
    idx = torch.nonzero(bad).flatten()
    print("worst violation:", float((err/tol).max()))
    for i in idx[:10]:
        print(float(vals[i]), float(out[i]), float(ref[i]), float(err[i]), float(tol[i]))
else:
    print("all pass; max err/tol ratio:", float((err/tol).max()))

print(json.dumps({"metric":"abs err vs contract tol, exhaustive FP16 in [-10,10]",
  "n":n, "violations":int(bad.sum()), "max_ratio":float((err/tol).max())}))
