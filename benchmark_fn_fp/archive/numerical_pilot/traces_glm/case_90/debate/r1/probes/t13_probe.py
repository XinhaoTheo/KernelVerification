import sys, json
sys.path.insert(0, "/root/pilot_cases/case_90")
import torch, numpy as np
import kernel as K

a, b = K.make_inputs("cuda")
out = K.run(a, b).cpu().numpy()
finite_count = int(np.isfinite(out).sum())
total = out.size
nonfinite = int(total - finite_count)
print(json.dumps({
    "finite": bool(finite_count == total),
    "finite_count": finite_count, "total": total, "nonfinite_count": nonfinite,
    "max_abs_out": float(np.abs(out).max()), "min_abs_out": float(np.abs(out).min()),
    "has_nan": bool(np.isnan(out).any()), "has_inf": bool(np.isinf(out).any()),
}))