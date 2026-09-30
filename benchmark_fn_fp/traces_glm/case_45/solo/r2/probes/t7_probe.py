import sys, json
sys.path.insert(0, "/root/numerical_cases/case_j")
import torch, numpy as np
from kernel import make_inputs, run

x = make_inputs()[0]
out = run(x)

xf = x.to(torch.float64)
mean = xf.sum() / 128
var = ((xf - mean) ** 2).sum() / 128
ref = (xf - mean) / torch.sqrt(var + 1e-5)

num = (out.to(torch.float64) - ref).norm().item()
den = max(ref.norm().item(), 0.001 * np.sqrt(128))
err = num / den
res = {"shape": list(out.shape), "dtype": str(out.dtype), "finite": bool(torch.isfinite(out).all().item()),
       "rel_l2_error": err, "bound": 0.02, "passes": bool(err <= 0.02),
       "max_abs_err": (out.to(torch.float64) - ref).abs().max().item()}
print(json.dumps(res))
