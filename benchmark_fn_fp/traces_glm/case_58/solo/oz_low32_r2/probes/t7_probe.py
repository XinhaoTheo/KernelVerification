import json, numpy as np, torch
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/numerical_cases/case_w/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
m = k.make_inputs("cuda")[0]
out = k.run(m)
torch.cuda.synchronize()
val = float(out[0].item())
A64 = m.cpu().numpy().astype(np.float64)
ref = float(np.log(np.linalg.det(A64)))
err = abs(val - ref) / max(abs(ref), 1.0)
print(json.dumps({"output": val, "reference": ref, "metric": "relative_error", "error": err,
 "shape": list(out.shape), "dtype": str(out.dtype), "finite": bool(np.isfinite(val)), "pass": err <= 1e-4}))
