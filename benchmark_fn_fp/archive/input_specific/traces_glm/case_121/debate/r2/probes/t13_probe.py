import numpy as np, torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_q/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
q, c, e = k.make_inputs()
out = k.run(q, c, e)
d = ((c.double() - q.double())**2).sum(dim=1)
ref_winner = int(torch.argmin(d).item())
ref = e[ref_winner].double()
rel = float((out.double() - ref).norm() / max(ref.norm().item(), 1e-12))
print(json.dumps({"ref_winner": ref_winner, "rel_l2_error": rel,
  "tolerance": 0.1, "passes_contract": rel <= 0.1,
  "ref_norm": float(ref.norm()),
  "out": out.tolist(), "ref": ref.tolist()}))