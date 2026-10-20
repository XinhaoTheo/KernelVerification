
import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_85/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, w = k.make_inputs("cuda")
out = k.run(x, w).cpu().double().numpy()
ref = w.cpu().double().numpy() @ x.cpu().double().numpy()
n = ref.size
E = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(n))
print(json.dumps({"E": float(E), "finite": bool(np.all(np.isfinite(out))), "ref_norm": float(np.linalg.norm(ref)), "out_norm": float(np.linalg.norm(out))}))
