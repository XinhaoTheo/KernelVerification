import torch, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/pilot_cases/case_102/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
q, k, v = m.make_inputs(device="cuda")
out = m.run(q, k, v)
# float64 reference on the same float32 input values
q64 = q.cpu().numpy().astype(np.float64)
k64 = k.cpu().numpy().astype(np.float64)
v64 = v.cpu().numpy().astype(np.float64)
z = k64 @ q64 / np.sqrt(32)
e = np.exp(z - z.max()); p = e / e.sum()
ref = p @ v64
o = out.cpu().numpy().astype(np.float64)
err = np.linalg.norm(o - ref); refn = np.linalg.norm(ref)
denom = max(refn, 0.001*np.sqrt(32))
E = err / denom
# also isolate the fp16 p-quantization term
p16 = p.astype(np.float16).astype(np.float64)
p16n = p16 / p16.sum()  # note kernel does NOT renormalize after cast
err_iso = np.linalg.norm((p16 - p16n*0) * 0) # placeholder
dy = np.linalg.norm((p16) @ v64 - ref)
print(json.dumps({
  "err_norm": float(err), "ref_norm": float(refn), "floor": 0.001*np.sqrt(32),
  "E": float(E), "E_le_0.001": bool(E <= 0.001),
  "fp16_p_error_only_norm": float(dy),
  "finite": bool(np.all(np.isfinite(o))),
  "p_sum_after_fp16": float(p16.sum()),
}))