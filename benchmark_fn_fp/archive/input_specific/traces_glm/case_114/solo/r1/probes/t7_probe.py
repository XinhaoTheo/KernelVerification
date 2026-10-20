import numpy as np, torch, sys, os
sys.path.insert(0, "/root/numerical_cases/case_p")
from kernel import run, make_inputs_numpy

u_np, b_np = make_inputs_numpy()
u = torch.from_numpy(u_np).to("cuda")
b = torch.from_numpy(b_np).to("cuda")
out = run(u, b).cpu().numpy().astype(np.float64)

# float64 reference on actual stored inputs, with recentring (b stored = round(1.125u + eps))
u64 = u_np.astype(np.float64); b64 = b_np.astype(np.float64)
alpha = (u64*b64).sum() / (u64*u64).sum()
res = b64 - alpha*u64
ref = res / np.linalg.norm(res)
rel = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-12)
print(np.round(float(rel), 6), bool(np.isfinite(out).all()))
