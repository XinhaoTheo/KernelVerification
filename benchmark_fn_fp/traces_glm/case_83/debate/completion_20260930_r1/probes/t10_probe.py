import json, sys, types
import numpy as np, torch

sys.path.insert(0, "/root/pilot_cases/case_83")
import kernel as K

a, b = K.make_inputs()
out = K.run(a, b)
out32 = out.detach().cpu().numpy().astype(np.float64)
a64 = a.detach().cpu().numpy().astype(np.float64)
b64 = b.detach().cpu().numpy().astype(np.float64)

# float64 reference on same inputs
ref = np.empty_like(b64)
h = np.zeros(b64.shape[1], dtype=np.float64)
for t in range(b64.shape[0]):
    h = a64[t] * h + b64[t]
    ref[t] = h

numel = ref.size
denom = max(np.linalg.norm(ref), 0.001 * np.sqrt(numel))
err = out32 - ref
E = np.linalg.norm(err) / denom

# error growth over time (per-t row RMS error and relative to row norms)
row_err = np.sqrt((err ** 2).mean(axis=1))
row_ref = np.sqrt((ref ** 2).mean(axis=1))
growth = [(t, float(row_err[t]), float(row_err[t] / row_ref[t])) for t in (0, 63, 159, 319, 479, 639)]
rel_tail = float(np.linalg.norm(err[576:]) / np.linalg.norm(ref[576:]))

print(json.dumps({
    "E": float(E), "budget": 0.003, "passes": bool(E <= 0.003),
    "finite": bool(np.isfinite(out32).all()),
    "ref_norm": float(np.linalg.norm(ref)), "denom": float(denom),
    "err_norm": float(np.linalg.norm(err)),
    "max_abs_err": float(np.abs(err).max()),
    "rel_err_tail_576_639": rel_tail,
    "growth": growth,
    "shape": list(out32.shape),
}, default=float))
