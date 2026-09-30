import numpy as np, torch, json, sys, os
sys.path.insert(0, "/root/pilot_cases/case_104")
os.chdir("/root/pilot_cases/case_104")
import kernel as K

dev = "cuda"
x, w = K.make_inputs(dev)
out = K.run(x, w)
out = out.float().cpu().numpy().astype(np.float64)
xf = x.cpu().numpy().astype(np.float64)
wf = w.cpu().numpy().astype(np.float64)
ref = wf @ xf
n = ref.size
err = out - ref
E = np.linalg.norm(err) / max(np.linalg.norm(ref), 0.001*np.sqrt(n))
finite = bool(np.all(np.isfinite(out)))
print(json.dumps({"E": float(E), "budget": 0.12, "all_finite": finite,
                  "ref_norm": float(np.linalg.norm(ref)),
                  "err_norm": float(np.linalg.norm(err)),
                  "n_out": int(n), "E_over_budget": bool(E > 0.12)}))
