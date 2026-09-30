import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_k/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
m_np, r_np = k.make_inputs_numpy()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m, r = k.make_inputs(dev)
out = k.run(m, r)
x_ref = np.linalg.solve(m_np.astype(np.float64), r_np.astype(np.float64))
out_np = out.cpu().numpy().astype(np.float64)
err = np.linalg.norm(out_np - x_ref) / max(np.linalg.norm(x_ref), 0.004)
print(json.dumps({"rel_l2_error": float(err), "tolerance": 0.08, "norm_x": float(np.linalg.norm(x_ref)), "finite": bool(np.all(np.isfinite(out_np))), "device": dev}))