import sys, json
sys.path.insert(0, "/root/numerical_cases/case_i")
import numpy as np, torch, kernel

(x,) = kernel.make_inputs("cuda")
out = kernel.run(x)
torch.cuda.synchronize()

xd = x.cpu().numpy().astype(np.float64)
mean = xd.sum() / 128
var = ((xd - mean) ** 2).sum() / 128
ref = (xd - mean) / np.sqrt(var + 1e-5)

o = out.cpu().numpy().astype(np.float64)
err = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 0.001 * np.sqrt(128))
print(json.dumps({
    "metric": "relative L2 error (contract formula)",
    "error": float(err),
    "threshold": 0.02,
    "shape": list(out.shape),
    "finite": bool(np.isfinite(o).all()),
    "dtype": str(out.dtype),
    "max_abs_err": float(np.abs(o - ref).max()),
}))
