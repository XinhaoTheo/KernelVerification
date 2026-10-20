import sys, json
sys.path.insert(0, "/root/pilot_cases/case_99")
import torch, numpy as np
from kernel import run, make_inputs

q, k, v = make_inputs("cuda")
out = run(q, k, v)
torch.cuda.synchronize()
out = out.float()

# float64 reference on same float32 input values
q64 = q.double().cpu().numpy(); k64 = k.double().cpu().numpy(); v64 = v.double().cpu().numpy()
z = k64 @ q64 / np.sqrt(q64.shape[0])
e = np.exp(z - z.max()); p = e / e.sum()
ref = p @ v64

diff = out.cpu().numpy().astype(np.float64) - ref
denom = max(np.linalg.norm(ref), 0.001*np.sqrt(ref.size))
E = np.linalg.norm(diff) / denom
finite = bool(np.all(np.isfinite(out.cpu().numpy())))
print(json.dumps({"E": float(E), "finite": finite, "ref_norm": float(np.linalg.norm(ref)),
                  "max_abs_err": float(np.abs(diff).max()), "shape": list(ref.shape),
                  "passes": bool(finite and E <= 0.001)}))
