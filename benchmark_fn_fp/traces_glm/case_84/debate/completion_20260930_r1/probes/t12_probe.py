import json, sys, math
sys.path.insert(0, "/root/pilot_cases/case_84")
import torch, numpy as np
import kernel as K

q, k, v = K.make_inputs("cuda")
out = K.run(q, k, v).double().cpu().numpy()

# float64 reference on the same float32 inputs
q64 = q.double().cpu().numpy(); k64 = k.double().cpu().numpy(); v64 = v.double().cpu().numpy()
z = k64 @ q64 / math.sqrt(q64.shape[0])
z = z - z.max(); p = np.exp(z); p /= p.sum()
ref = p @ v64

err = np.linalg.norm(out - ref)
refn = np.linalg.norm(ref)
denom = max(refn, 0.001*math.sqrt(32))
E = err/denom
print(json.dumps({
    "metric": "contract E on exact workload",
    "E": E, "err_l2": err, "ref_norm": refn, "denominator": denom,
    "floor": 0.001*math.sqrt(32),
    "finite": bool(np.isfinite(out).all()),
    "E_le_0.001": bool(E <= 0.001)
}))