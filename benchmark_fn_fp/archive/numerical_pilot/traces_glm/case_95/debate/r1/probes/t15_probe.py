import sys, json
sys.path.insert(0, "/root/pilot_cases/case_95")
import torch, numpy as np
from kernel import make_inputs, run

x, w = make_inputs("cuda")
out = run(x, w)
xn = x.detach().cpu().numpy().astype(np.float64)
wn = w.detach().cpu().numpy().astype(np.float64)
ref = wn @ xn
o = out.detach().cpu().numpy().astype(np.float64)
den = max(np.linalg.norm(ref), 0.001*np.sqrt(ref.size))
err = o - ref
E = np.linalg.norm(err) / den
# decomposition: coherent residual-aligned component
scale = np.max(np.abs(wn), axis=1, keepdims=True)/7.0
q = np.clip(np.floor(wn/scale + 0.5), -7, 7)*scale
R = q - wn                     # (32,128) per-row residual
r_col = R.sum(axis=0); r_col /= np.linalg.norm(r_col)
# per-row error contribution from residual-aligned part of x:
# err_i ~ x . r_i ; aligned part uses 0.5*r_col direction
aligned = 0.5 * (R @ r_col)    # (32,)
print(json.dumps({"E": float(E), "budget": 0.12, "E_pass": bool(E <= 0.12),
 "finite": bool(np.all(np.isfinite(o))),
 "ref_norm": float(np.linalg.norm(ref)), "err_norm": float(np.linalg.norm(err)),
 "aligned_err_norm": float(np.linalg.norm(aligned)),
 "x_dot_r_col": float(xn @ r_col)}))