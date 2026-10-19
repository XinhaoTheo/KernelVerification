
import numpy as np, torch, math, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("kern", "/root/numerical_cases/case_j/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, = k.make_inputs(device="cuda")
out = k.run(x)
x64 = x.cpu().numpy().astype(np.float64)
mean = x64.sum()/128
var = ((x64-mean)**2).sum()/128
ref = (x64-mean)/np.sqrt(var+1e-5)
o = out.cpu().numpy().astype(np.float64)
num = np.linalg.norm(o-ref)
den = max(np.linalg.norm(ref), 0.001*math.sqrt(128))
rel = num/den
print(json.dumps({"rel_l2": float(rel), "mean": mean, "var": var, "num": float(num),
                  "finite": bool(np.isfinite(o).all()), "shape": list(out.shape)}))
