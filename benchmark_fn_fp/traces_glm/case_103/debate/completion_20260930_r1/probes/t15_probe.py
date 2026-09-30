import sys, json, torch, numpy as np
sys.path.insert(0, "/root/pilot_cases/case_103")
import kernel as K

q, k, v = K.make_inputs("cuda")
out = K.run(q, k, v)
torch.cuda.synchronize()

# float64 reference on the SAME float32 input values
q64, k64, v64 = q.double(), k.double(), v.double()
z = k64 @ q64 / np.sqrt(32.0)
z = z - z.max()
p = torch.exp(z)
p = p / p.sum()
ref = p @ v64

err = (out.double() - ref)
err_norm = float(err.norm().item())
ref_norm = float(ref.norm().item())
floor = 0.001 * float(np.sqrt(32.0))
E = err_norm / max(ref_norm, floor)
result = {
    "E": E,
    "budget": 0.001,
    "E_exceeds_budget": bool(E > 0.001),
    "ref_norm": ref_norm,
    "denominator_floor": floor,
    "floor_engaged": bool(ref_norm < floor),
    "err_norm": err_norm,
    "out_finite": bool(torch.isfinite(out).all().item()),
    "max_abs_err": float(err.abs().max().item()),
    "device": str(out.device),
    "E_within_budget": bool(E <= 0.001),
}
print(json.dumps(result))