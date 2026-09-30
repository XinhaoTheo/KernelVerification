import json, sys
import importlib.util, torch
spec = importlib.util.spec_from_file_location("kernel", "/root/pilot_cases/case_92/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, b = k.make_inputs("cuda")
out = k.run(a, b)
print(json.dumps({
    "all_finite": bool(torch.isfinite(out).all().item()),
    "nonfinite_count": int((~torch.isfinite(out)).sum().item()),
    "max_abs": float(out.abs().max()), "min_abs": float(out.abs().min()),
    "shape": list(out.shape), "dtype": str(out.dtype)}))