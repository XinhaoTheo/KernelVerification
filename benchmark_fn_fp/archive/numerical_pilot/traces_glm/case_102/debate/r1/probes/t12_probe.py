import torch, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/pilot_cases/case_102/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
q, k, v = m.make_inputs(device="cuda")
out = m.run(q, k, v)
q64 = q.cpu().numpy().astype(np.float64)
k64 = k.cpu().numpy().astype(np.float64)
v64 = v.cpu().numpy().astype(np.float64)
z = k64 @ q64 / np.sqrt(32)
e = np.exp(z - z.max()); p = e / e.sum()
ref = p @ v64
o = out.cpu().numpy().astype(np.float64)
err = float(np.linalg.norm(o - ref)); refn = float(np.linalg.norm(ref))
budget = 0.001 * max(refn, 0.001*np.sqrt(32))
print(json.dumps({
  "abs_err_norm": err, "abs_budget": float(budget),
  "exceeds_budget": bool(err > budget),
  "ref_norm": refn, "floor": 0.001*np.sqrt(32),
  "regime": "relative" if refn > 0.001*np.sqrt(32) else "floor-dominated",
  "fp16_p_l1_quant_err": float(np.abs(p.astype(np.float16).astype(np.float64) - p).sum()),
}))