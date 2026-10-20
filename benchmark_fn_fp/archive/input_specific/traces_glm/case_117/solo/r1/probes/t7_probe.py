import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_s")
import kernel

amp, freq, ph = kernel.make_inputs()
out = kernel.run(amp, freq, ph).cpu().numpy()

a = amp.cpu().numpy().astype(np.float64)
f = freq.cpu().numpy().astype(np.float64)
p = ph.cpu().numpy().astype(np.float64)
ref = 1.0 + (a * (np.cos(p) - np.cos(p + f)) / f).sum(axis=1)

err = np.linalg.norm(out.astype(np.float64) - ref) / max(np.linalg.norm(ref), 1e-12)
print(json.dumps({
    "output": out.tolist(), "reference": ref.tolist(),
    "rel_l2": err, "tolerance": 0.035,
    "max_freq": float(f.max()), "finite": bool(np.isfinite(out).all()),
    "passes_contract": bool(err <= 0.035)
}))