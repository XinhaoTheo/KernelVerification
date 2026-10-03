import json, importlib.util, torch, numpy as np
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_100/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
x, w = mod.make_inputs("cuda")
out = mod.run(x, w).double()
ref = (w.double() @ x.double())
n = ref.numel()
den = max(float(ref.norm()), 0.001*np.sqrt(n))
E = float((out - ref).norm()) / den
print(json.dumps({
  "E": E, "threshold": 0.12, "margin": 0.12 - E,
  "err_norm": float((out-ref).norm()), "ref_norm": float(ref.norm()), "denom": den,
  "finite": bool(torch.isfinite(out).all()), "E_le_0.12": E <= 0.12,
  "max_abs_err": float((out-ref).abs().max()),
  "max_rel_err": float(((out-ref).abs()/ref.abs().clamp_min(1e-12)).max())
}))