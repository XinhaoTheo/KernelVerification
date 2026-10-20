import json, sys, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/numerical_cases/case_h/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
values = k.make_inputs("cuda")[0]
out = k.run(values)
ref = values.to(torch.float64).sum(dim=1) if False else None
import torch
ref = values.double().sum(dim=1)
diff = (out.double() - ref)
E = diff.norm().item() / max(ref.norm().item(), 0.008)
print(json.dumps({"E": E, "finite": bool(torch.isfinite(out).all().item()),
 "max_abs_err": diff.abs().max().item(), "ref_norm": ref.norm().item(),
 "sample_out": out[:4].tolist(), "sample_ref": ref[:4].tolist()}))
