
import numpy as np, torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_l/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

m_np, rhs_np = k.make_inputs_numpy()
m64 = m_np.astype(np.float64); r64 = rhs_np.astype(np.float64)
x = np.linalg.solve(m64, r64)  # float64 reference

m, rhs = k.make_inputs("cuda")
out = k.run(m, rhs)
out_np = out.detach().cpu().numpy().astype(np.float64)
den = max(np.linalg.norm(x), 0.001*np.sqrt(16))
err = np.linalg.norm(out_np - x) / den
print(json.dumps({"rel_l2_error": float(err), "bound": 0.08, "finite": bool(np.all(np.isfinite(out_np))), "norm_x": float(np.linalg.norm(x)), "output_first4": out_np[:4].tolist()}))
