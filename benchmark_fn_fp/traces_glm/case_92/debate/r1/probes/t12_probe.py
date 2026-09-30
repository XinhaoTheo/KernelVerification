import json, sys
sys.path.insert(0, "/root/pilot_cases/case_92")
import importlib.util, torch
spec = importlib.util.spec_from_file_location("kernel", "/root/pilot_cases/case_92/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, b = k.make_inputs("cuda")
out = k.run(a, b)
# fp64 reference on the same fp32 input values
af = a.double().cpu(); bf = b.double().cpu()
ref = torch.empty_like(bf)
h = torch.zeros(af.shape[1], dtype=torch.float64)
for t in range(af.shape[0]):
    h = af[t] * h + bf[t]
    ref[t] = h
o = out.double().cpu()
n = ref.numel()
E = (o - ref).norm() / max(ref.norm(), 0.001 * (n ** 0.5))
print(json.dumps({
    "E": float(E), "budget": 0.003, "E_within_budget": bool(E <= 0.003),
    "ref_norm": float(ref.norm()), "err_norm": float((o - ref).norm()),
    "max_abs_err": float((o - ref).abs().max()), "all_finite": bool(torch.isfinite(out).all().item()),
    "shape": list(out.shape), "dtype": str(out.dtype)}))