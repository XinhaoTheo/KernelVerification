import sys, os, json
sys.path.insert(0, "/root/evidence_cases/case_e08")
import numpy as np, itertools, torch
import kernel as K

features, anchors = K.make_inputs()
f0 = features.cpu().numpy().astype(np.float64)
f0_copy = f0.copy()
out = K.run(features, anchors)
y = out.detach().cpu().numpy().astype(np.float64)
inputs_unmodified = bool(np.array_equal(f0, features.cpu().numpy().astype(np.float64)))

denom = max(np.linalg.norm(f0), 1e-12)
best = float("inf"); best_t = None
for p in itertools.permutations(range(4)):
    for s in itertools.product((-1, 1), repeat=4):
        expected = f0[:, :, list(p)] * np.asarray(s)
        e = float(np.linalg.norm(y - expected) / denom)
        if e < best:
            best = e; best_t = {"permutation": list(p), "signs": list(s)}
# also record where column 1 lands per view in the output (nonzero argmax col)
cols_v0 = [int(np.argmax(np.abs(y[0]).sum(0)))] 
result = {
    "metric": "joint shared-transform E over 384 (p,s), float64, fixed seed-171200 workload",
    "best_E": best,
    "best_transform": best_t,
    "tolerance": 0.05,
    "passes_E": bool(best <= 0.05),
    "inputs_unmodified": inputs_unmodified,
    "output_shape": list(y.shape),
    "output_finite": bool(np.isfinite(y).all()),
    "anchor_values": anchors.cpu().numpy().tolist(),
    "anchor_abs": np.abs(anchors.cpu().numpy()).astype(np.float64).tolist(),
}
print(json.dumps(result))
