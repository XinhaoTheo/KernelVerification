import json, numpy as np, torch
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/evidence_cases/case_e13/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
res = k.initial_probe()
x, = k.make_inputs()
out = k.run(x)
vals64 = x.cpu().numpy().astype(np.float64)
target = vals64.sum(axis=1)  # exact: 2^80 + ... - 2^80 cancels
rel = np.linalg.norm(out.cpu().numpy().astype(np.float64)-target)/max(np.linalg.norm(target),1e-12)
print(json.dumps({"probe_result": res, "exact_target": target.tolist(), "output": out.cpu().numpy().tolist(), "relative_error_vs_exact": float(rel), "tolerance": 1e-5, "metric": "relative L2 vs exact float64 row sums on the fixed make_inputs workload"}))