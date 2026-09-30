import json, numpy as np, torch, importlib.util
spec = importlib.util.spec_from_file_location("kernel_mod", "/root/pilot_cases/case_88/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
q, k, v = m.make_inputs("cuda")
out = m.run(q, k, v)
ref = torch.zeros(32, dtype=torch.float64)
q64 = q.to(torch.float64); k64 = k.to(torch.float64); v64 = v.to(torch.float64)
z = (k64 @ q64) / np.sqrt(32.0)
p = torch.exp(z - z.max()); p = p / p.sum()
ref = p @ v64
nelem = out.numel()
E = (out.to(torch.float64) - ref).norm().item() / max(ref.norm().item(), 0.001*np.sqrt(nelem))
print(json.dumps({"E": E, "finite": bool(torch.isfinite(out).all().item()), "ref_norm": ref.norm().item(), "out_norm": out.to(torch.float64).norm().item()}))