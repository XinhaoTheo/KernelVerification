import sys, json
sys.path.insert(0, "/root/numerical_cases/case_t")
import numpy as np, torch
import kernel as K

amp, freq, ph = K.make_inputs("cuda")
out = K.run(amp, freq, ph)
torch.cuda.synchronize()

a = amp.double().cpu().numpy(); f = freq.double().cpu().numpy(); p = ph.double().cpu().numpy()
ref = 1.0 + (a * (np.cos(p) - np.cos(p + f)) / f).sum(axis=1)
o = out.double().cpu().numpy()
rel = np.linalg.norm(o - ref) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({"output": o.tolist(), "reference": ref.tolist(),
                  "rel_l2": float(rel), "threshold": 0.035,
                  "passes": bool(rel <= 0.035), "finite": bool(np.all(np.isfinite(o)))}))