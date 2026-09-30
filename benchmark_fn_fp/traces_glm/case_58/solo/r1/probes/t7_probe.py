
import sys, json
sys.path.insert(0, "/root/numerical_cases/case_w")
import numpy as np, torch
import kernel

(matrix,) = kernel.make_inputs(device="cuda")
out = kernel.run(matrix)
torch.cuda.synchronize()

ref = np.linalg.slogdet(matrix.cpu().numpy().astype(np.float64))[1]
o = float(out.cpu().numpy()[0])
err = abs(o - ref) / max(abs(ref), 1.0)
print(json.dumps({
    "metric": "relative error vs float64 slogdet on fixed workload",
    "output": o, "reference": ref, "rel_err": err,
    "finite": bool(np.isfinite(o)), "dtype": str(out.dtype), "shape": list(out.shape),
    "tol": 1e-4, "pass": bool(err <= 1e-4 and np.isfinite(o)),
}))
