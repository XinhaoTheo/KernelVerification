import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_l/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m, b = k.make_inputs(dev)
out = k.run(m, b).cpu().numpy()
A = m.cpu().numpy().astype(np.float64); rhs = b.cpu().numpy().astype(np.float64)
x = np.linalg.solve(A, rhs)
state = np.zeros(16)
for _ in range(64):
    state = state + (rhs - A @ state)
den = max(np.linalg.norm(x), 0.04)
total = np.linalg.norm(out - x)/den
f64err = np.linalg.norm(state - x)/den
print(json.dumps({"device": dev, "rel_err_kernel_vs_f64_reference": float(total),
                  "rel_err_f64_64step_vs_reference": float(f64err),
                  "fp32_rounding_extra": float(total - f64err),
                  "finite": bool(np.all(np.isfinite(out))), "tolerance": 0.08,
                  "x_norm": float(np.linalg.norm(x))}))