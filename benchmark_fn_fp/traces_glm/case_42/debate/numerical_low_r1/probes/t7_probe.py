
import sys, json
sys.path.insert(0, "/root/numerical_cases/case_g")
import numpy as np, torch
import kernel as K

(vals,) = K.make_inputs(device="cuda")
out = K.run(vals)
ref = vals.double().sum(dim=1)
err = (out.double() - ref)
E = err.norm().item() / max(ref.norm().item(), 0.008)
print(json.dumps({
  "E": E,
  "tolerance": 0.1,
  "passes_contract": bool(E <= 0.1 and torch.isfinite(out).all().item()),
  "max_abs_err": err.abs().max().item(),
  "ref_norm": ref.norm().item(),
  "ref_min": ref.min().item(), "ref_max": ref.max().item(),
  "finite": bool(torch.isfinite(out).all().item()),
}))
