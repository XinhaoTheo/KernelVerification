
import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_v")
import kernel

coeff_np, off_np = kernel.make_inputs_numpy()
coeff, off = kernel.make_inputs("cuda")
out = kernel.run(coeff, off)
torch.cuda.synchronize()
out_np = out.detach().cpu().numpy().astype(np.float64)

n = np.arange(64, dtype=np.float64)
ref = np.float64(off_np[0]) + np.zeros(64)
for k in range(1, 17):
    a = np.float64(coeff_np[0, k-1]); b = np.float64(coeff_np[1, k-1])
    ref += a*np.cos(2*np.pi*k*n/64) + b*np.sin(2*np.pi*k*n/64)

num = np.linalg.norm(out_np - ref)
den = max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
  "metric": "relative L2 error vs float64 16-mode reference",
  "rel_l2": num/den,
  "norm_diff": float(num), "norm_ref": float(den),
  "shape": list(out_np.shape), "dtype": str(out.dtype),
  "finite": bool(np.all(np.isfinite(out_np))),
  "tolerance": 0.15, "pass": bool(num/den <= 0.15)
}))
