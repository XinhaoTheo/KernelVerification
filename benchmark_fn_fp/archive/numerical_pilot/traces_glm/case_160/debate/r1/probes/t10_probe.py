import json, numpy as np, torch, sys
sys.path.insert(0, "/root/pilot_cases/case_91")
import kernel as K
x, w = K.make_inputs("cuda")
out = K.run(x, w)
ref = w.cpu().numpy().astype(np.float64) @ x.cpu().numpy().astype(np.float64)
o = out.cpu().numpy().astype(np.float64)
n = ref.size
E = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 0.001 * np.sqrt(n))
print(json.dumps({"E": float(E), "budget": 0.12, "all_finite": bool(np.all(np.isfinite(o))),
                  "ref_norm": float(np.linalg.norm(ref)), "max_abs_err": float(np.abs(o-ref).max()),
                  "m": 32, "k": 256, "passes": bool(E <= 0.12)}))
