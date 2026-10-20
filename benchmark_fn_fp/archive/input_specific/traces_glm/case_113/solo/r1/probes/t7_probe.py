import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_i")
import kernel as k

(x,) = k.make_inputs("cuda")
out = k.run(x)
x64 = x.cpu().numpy().astype(np.float64)
mean = x64.sum() / 128
var = ((x64 - mean) ** 2).sum() / 128
ref = (x64 - mean) / np.sqrt(var + 1e-5)
o = out.detach().cpu().numpy().astype(np.float64)
err = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 0.001 * np.sqrt(128))
print(json.dumps({
    "metric": "relative L2 error vs float64 reference",
    "error": float(err),
    "threshold": 0.02,
    "finite": bool(np.isfinite(o).all()),
    "shape": list(o.shape),
    "max_abs_diff": float(np.abs(o - ref).max()),
    "denominator_kernelish": float(np.sqrt(max(var, 0) + 1e-5)),
}))
