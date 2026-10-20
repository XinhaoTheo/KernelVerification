
import json, torch, numpy as np, sys
sys.path.insert(0, "/root/pilot_cases/case_95")
import kernel

x, w = kernel.make_inputs("cuda")
out = kernel.run(x, w)
ref = w.double() @ x.double()
diff = out.double() - ref
E = diff.norm().item() / max(ref.norm().item(), 0.001*np.sqrt(ref.numel()))
print(json.dumps({"E": E, "finite": bool(torch.isfinite(out).all().item()),
                  "ref_norm": ref.norm().item(), "max_abs_err": diff.abs().max().item()}))
