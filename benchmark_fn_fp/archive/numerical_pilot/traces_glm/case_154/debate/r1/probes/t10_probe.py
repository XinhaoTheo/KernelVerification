import sys, json, numpy as np, torch
sys.path.insert(0, "/root/pilot_cases/case_97")
import kernel as K

a, b = K.make_inputs(device="cuda")
out = K.run(a, b)

# float64 reference on same float32 inputs
a64 = a.cpu().numpy().astype(np.float64)
b64 = b.cpu().numpy().astype(np.float64)
t, d = a64.shape
ref = np.empty_like(b64)
h = np.zeros(d)
for i in range(t):
    h = a64[i] * h + b64[i]
    ref[i] = h

o = out.cpu().numpy().astype(np.float64)
N = o.size
num = np.linalg.norm((o - ref).ravel())
den = max(np.linalg.norm(ref.ravel()), 0.001 * np.sqrt(N))
E = num / den
finite = bool(np.isfinite(o).all())
diff = (o - ref).ravel()
print(json.dumps({
    "E": float(E),
    "budget": 0.003,
    "finite": finite,
    "rel_E_vs_budget": float(E / 0.003),
    "ref_norm": float(np.linalg.norm(ref.ravel())),
    "floor": float(0.001 * np.sqrt(N)),
    "max_abs_err": float(np.abs(diff).max()),
    "rms_abs_err": float(np.sqrt((diff**2).mean())),
    "tail_rms_abs_err": float(np.sqrt((diff[-100:]**2).mean())),
    "N": int(N),
}))
