import json, numpy as np, torch, sys
sys.path.insert(0, '/root/numerical_cases/case_f')
from kernel import run, make_inputs, make_inputs_numpy

matrix, initial, drive = make_inputs()
out = run(matrix, initial, drive)
torch.cuda.synchronize()

m64 = matrix.cpu().numpy().astype(np.float64)
s64 = initial.cpu().numpy().astype(np.float64)
d64 = drive.cpu().numpy().astype(np.float64)
for t in range(64):
    s64 = m64 @ s64 + d64[t]

o = out.cpu().numpy().astype(np.float64)
num = np.linalg.norm(o - s64)
den = max(np.linalg.norm(s64), 0.001*np.sqrt(16))
print(json.dumps({
    "metric": "relative L2 error vs float64 reference",
    "num": float(num), "den": float(den),
    "rel_err": float(num/den),
    "bound": 0.002,
    "finite": bool(np.all(np.isfinite(o))),
    "shape": list(o.shape),
    "passes": bool(num/den <= 0.002)
}))
