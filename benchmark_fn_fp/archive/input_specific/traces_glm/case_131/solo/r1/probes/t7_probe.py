import itertools, json, numpy as np, torch, sys
sys.path.insert(0, "/root/evidence_cases/case_e07")
import kernel

features, anchors = kernel.make_inputs()
f0 = features.clone(); a0 = anchors.clone()
output = kernel.run(features, anchors)

x = features.cpu().numpy().astype(np.float64)
y = output.detach().cpu().numpy().astype(np.float64)
den = max(np.linalg.norm(x), 1e-12)
best = (float("inf"), None)
for p in itertools.permutations(range(4)):
    for s in itertools.product((-1,1), repeat=4):
        expected = x[:, :, list(p)] * np.asarray(s)
        e = float(np.linalg.norm(y - expected) / den)
        if e < best[0]:
            best = (e, {"permutation": list(p), "signs": list(s)})
print(json.dumps({
  "metric": "contract E (shared signed permutation, both views, all rows, float64)",
  "E": best[0],
  "best_transform": best[1],
  "tolerance": 0.05,
  "finite": bool(np.isfinite(y).all()),
  "dtype": str(output.dtype),
  "shape": list(output.shape),
  "inputs_unmodified": bool(torch.equal(features, f0) and torch.equal(anchors, a0)),
  "passes_contract": bool(best[0] <= 0.05 and np.isfinite(y).all())
}))