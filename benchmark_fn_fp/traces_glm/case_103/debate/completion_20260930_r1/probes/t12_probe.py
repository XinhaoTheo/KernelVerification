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
E = err.norm().item() / max(ref.norm().item(), 0.001 * np.sqrt(32.0))
result = {
    "E": E,
    "budget": 0.001,
    "E_exceeds_budget": E > 0.001,
    "ref_norm": ref.norm().item(),
    "denominator_floor": 0.001 * np.sqrt(32.0),
    "floor_engaged": ref.norm().item() < 0.001 * np.sqrt(32.0),
    "err_norm": err.norm().item(),
    "out_finite": bool(torch.isfinite(out).all().item()),
    "max_abs_err": err.abs().max().item(),
    "device": str(out.device),
}
print(json.dumps(result))