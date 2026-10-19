import torch, numpy as np, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_86/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
x, w = k.make_inputs("cuda")
out = k.run(x, w)
ref = (w.double() @ x.double())
E = (out.double() - ref).norm() / max(ref.norm().item(), 0.001*np.sqrt(32))
print(json.dumps({"E": E.item(), "budget": 0.12, "ref_norm": ref.norm().item(), "finite": bool(torch.isfinite(out).all().item()), "max_abs_err": (out.double()-ref).abs().max().item()}))