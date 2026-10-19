
import numpy as np, json

# Fixed inputs exactly as in kernel.py make_inputs_numpy()
rng = np.random.Generator(np.random.PCG64(119130))
query = (16.0 + rng.normal(0.0, 0.5, 32)).astype(np.float32)
anchors = (query.astype(np.float64)[None, :] +
           rng.normal(0.0, 0.015625, (16, 32))).astype(np.float32)
values = rng.normal(0.0, 1.0, 16).astype(np.float32)

# float64 reference per contract
a64 = anchors.astype(np.float64); q64 = query.astype(np.float64); v64 = values.astype(np.float64)
d_ref = ((a64 - q64[None, :])**2).sum(axis=1)
w_ref = np.exp(-16.0 * d_ref)
ref = (w_ref * v64).sum() / w_ref.sum()

res = {"gpu_used": False, "kernel_output": None, "reference": float(ref), "metric": None, "error": None}
try:
    import torch, importlib.util
    spec = importlib.util.spec_from_file_location("kern", "/root/numerical_cases/case_y/kernel.py")
    kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
    q, an, va = kern.make_inputs()
    out = kern.run(q, an, va)
    ko = float(out.item())
    denom = max(abs(ref), 0.05)
    res.update(gpu_used=True, kernel_output=ko, metric=abs(ko - ref)/denom, device=str(q.device))
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"

print(json.dumps(res))
