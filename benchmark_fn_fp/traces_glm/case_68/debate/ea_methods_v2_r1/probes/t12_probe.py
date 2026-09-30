import sys, json, itertools
sys.path.insert(0, "/root/evidence_cases/case_e07")
import numpy as np, torch
from kernel import make_inputs, run

features, anchors = make_inputs()
f0 = features.clone(); a0 = anchors.clone()
out = run(features, anchors)
x = features.cpu().numpy().astype(np.float64)
y = out.detach().cpu().numpy().astype(np.float64)
den = max(np.linalg.norm(x), 1e-12)
best = {"E": float("inf"), "perm": None, "signs": None}
for p in itertools.permutations(range(4)):
    for s in itertools.product((-1, 1), repeat=4):
        expected = x[:, :, list(p)] * np.asarray(s)
        E = float(np.linalg.norm(y - expected) / den)
        if E < best["E"]:
            best = {"E": E, "perm": list(p), "signs": list(s)}
result = {
    "metric": "min shared (p,s) relative Frobenius error over all 384 transforms (contract E), float64",
    "E": best["E"], "best_perm": best["perm"], "best_signs": best["signs"],
    "tolerance": 0.05, "E_le_tol": bool(best["E"] <= 0.05),
    "finite_output": bool(np.isfinite(y).all()),
    "output_dtype": str(out.dtype), "output_shape": list(out.shape),
    "inputs_unmodified": bool(torch.equal(features, f0) and torch.equal(anchors, a0)),
}
print(json.dumps(result))