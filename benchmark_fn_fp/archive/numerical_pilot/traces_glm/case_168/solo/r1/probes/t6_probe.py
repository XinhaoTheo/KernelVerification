import sys, json, torch, numpy as np
sys.path.insert(0, '/root/pilot_cases/case_105')
from kernel import make_inputs, run

x, w = make_inputs('cuda')
out = run(x, w)
ref = w.detach().cpu().numpy().astype(np.float64) @ x.detach().cpu().numpy().astype(np.float64)
o = out.detach().cpu().numpy().astype(np.float64)
diff = o - ref
n = ref.size
E = np.linalg.norm(diff) / max(np.linalg.norm(ref), 0.001*np.sqrt(n))
print(json.dumps({"E": float(E), "finite": bool(np.isfinite(o).all()), "ref_norm": float(np.linalg.norm(ref)), "diff_norm": float(np.linalg.norm(diff)), "max_abs_err": float(np.abs(diff).max())}))