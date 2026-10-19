
import sys, json, numpy as np, torch
sys.path.insert(0, "/root/pilot_cases/case_104")
import kernel

x, w = kernel.make_inputs("cuda")
out = kernel.run(x, w)
ref = w.detach().cpu().numpy().astype(np.float64) @ x.detach().cpu().numpy().astype(np.float64)
o = out.detach().cpu().numpy().astype(np.float64)
n = ref.size
E = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 0.001 * np.sqrt(n))
print(json.dumps({"finite": bool(np.isfinite(o).all()), "E": float(E), "budget": 0.12, "ref_norm": float(np.linalg.norm(ref)), "out_norm": float(np.linalg.norm(o)), "n": int(n)}))
