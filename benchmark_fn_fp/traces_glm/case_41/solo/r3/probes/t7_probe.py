import torch, numpy as np, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_f/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
m, init, drive = k.make_inputs()
out = k.run(m, init, drive)
# float64 reference on the float32 inputs
M = m.to(torch.float64).cpu().numpy()
s = init.to(torch.float64).cpu().numpy().astype(np.float64)
D = drive.to(torch.float64).cpu().numpy().astype(np.float64)
for t in range(64):
    s = M @ s + D[t]
o = out.to(torch.float64).cpu().numpy()
num = np.linalg.norm(o - s)
den = max(np.linalg.norm(s), 0.001*np.sqrt(16))
print(json.dumps({
  "metric": "relative L2 error vs float64 reference",
  "reason": "contract tolerance is 0.002 on this exact workload",
  "rel_l2": num/den,
  "norm_out": float(np.linalg.norm(o)),
  "norm_ref": float(np.linalg.norm(s)),
  "finite": bool(np.all(np.isfinite(o))),
  "shape": list(o.shape),
  "max_abs_err": float(np.max(np.abs(o - s))),
  "pass": bool(num/den <= 0.002),
}))