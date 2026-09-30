import sys, json
sys.path.insert(0, "/root/pilot_cases/case_95")
import torch, numpy as np
from kernel import make_inputs, run

x, w = make_inputs("cuda")
out = run(x, w)
ref = w.detach().cpu().numpy().astype(np.float64) @ x.detach().cpu().numpy().astype(np.float64)
o = out.detach().cpu().numpy().astype(np.float64)
den = max(np.linalg.norm(ref), 0.001*np.sqrt(ref.size))
E = np.linalg.norm(o - ref) / den
# decompose: error vector of kernel vs quantized-fp64 prediction
wf = w.cpu().numpy().astype(np.float64)
scale = np.max(np.abs(wf), axis=1, keepdims=True)/7.0
q = np.clip(np.floor(wf/scale + 0.5), -7, 7)*scale
quant_ref = wf @ x.cpu().numpy().astype(np.float64)
r_col = (q - wf).sum(axis=0); r_col /= np.linalg.norm(r_col)
err = o - ref
print(json.dumps({"E": float(E), "budget": 0.12, "finite": bool(np.all(np.isfinite(o))),
 "ref_norm": float(np.linalg.norm(ref)), "err_norm": float(np.linalg.norm(err)),
 "err_aligned_with_r_col": float(err @ r_col), "E_pass": bool(E <= 0.12)}))