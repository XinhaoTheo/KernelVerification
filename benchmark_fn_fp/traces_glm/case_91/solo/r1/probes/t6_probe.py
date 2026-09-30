import json, sys, numpy as np, torch
sys.path.insert(0, "/root/pilot_cases/case_91")
import kernel

x, w = kernel.make_inputs("cuda")
out = kernel.run(x, w)
ref = w.to(torch.float64) @ x.to(torch.float64)
diff = out.to(torch.float64) - ref
E = float(diff.norm() / max(float(ref.norm()), 0.001*np.sqrt(ref.numel())))
print(json.dumps({
  "metric": "E budget ratio per problem.txt",
  "shapes": list(w.shape),
  "finite": bool(torch.isfinite(out).all()),
  "E": E,
  "budget": 0.12,
  "norm_ref": float(ref.norm()),
  "norm_diff": float(diff.norm()),
  "max_abs_err": float(diff.abs().max()),
  "passes": bool(torch.isfinite(out).all() and E <= 0.12),
}))
