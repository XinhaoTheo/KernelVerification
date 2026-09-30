import json, math, torch
import importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/pilot_cases/case_82/kernel.py")
kern = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kern)

q, k, v = kern.make_inputs(device="cuda")
out = kern.run(q, k, v)

# float64 reference on the exact float32 input values
q64 = q.to(torch.float64).cpu()
k64 = k.to(torch.float64).cpu()
v64 = v.to(torch.float64).cpu()
z = (k64 @ q64) / math.sqrt(32)
e = z - z.max()
p = torch.exp(e); p = p / p.sum()
ref = p @ v64

diff = out.to(torch.float64).cpu() - ref
num = diff.norm().item()
refn = ref.norm().item()
floor = 0.001 * math.sqrt(32)
den = max(refn, floor)
E = num / den
print(json.dumps({
    "E": E,
    "budget": 0.001,
    "E_exceeds_budget": bool(E > 0.001),
    "numerator_norm": num,
    "ref_norm": refn,
    "floor": floor,
    "floor_dominates": bool(refn < floor),
    "denominator": den,
    "all_finite": bool(torch.isfinite(out).all().item()),
    "max_abs_err": diff.abs().max().item(),
    "out_shape": list(out.shape),
}))