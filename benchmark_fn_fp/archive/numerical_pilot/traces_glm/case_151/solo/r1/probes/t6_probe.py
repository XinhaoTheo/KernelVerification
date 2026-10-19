import sys, json
sys.path.insert(0, "/root/pilot_cases/case_94")
import numpy as np, torch, math
import kernel as K

q, k, v = K.make_inputs("cuda")
out = K.run(q, k, v)

# float64 reference on the SAME float32 input values
q64 = q.cpu().numpy().astype(np.float64)
k64 = k.cpu().numpy().astype(np.float64)
v64 = v.cpu().numpy().astype(np.float64)
z = k64 @ q64 / math.sqrt(q64.shape[0])
z = z - z.max()
p = np.exp(z); p /= p.sum()
ref = p @ v64

o = out.cpu().numpy().astype(np.float64)
finite = bool(np.all(np.isfinite(o)))
num = np.linalg.norm(o - ref)
den = max(np.linalg.norm(ref), 0.001 * math.sqrt(ref.size))
E = num / den
print(json.dumps({"E": E, "finite": finite, "num": num,
                  "ref_norm": float(np.linalg.norm(ref)),
                  "out_norm": float(np.linalg.norm(o)),
                  "shape": list(o.shape)}))
