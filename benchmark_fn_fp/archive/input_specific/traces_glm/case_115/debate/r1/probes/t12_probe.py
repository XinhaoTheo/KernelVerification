import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_q")
from kernel import run, make_inputs

query, candidates, embeddings = make_inputs()
out = run(query, candidates, embeddings)

# exact float64 reference
q = query.to(torch.float64).numpy()
c = candidates.to(torch.float64).numpy()
d = ((c - q) ** 2).sum(axis=1)
ref_idx = int(np.argmin(d))  # argmin returns first (smallest) index on ties
ref_out = embeddings[ref_idx].numpy().astype(np.float64)
rel = np.linalg.norm(out.numpy().astype(np.float64) - ref_out) / max(np.linalg.norm(ref_out), 1e-12)

print(json.dumps({
    "metric": "kernel winner vs exact float64 winner on fixed seed-840104 workload",
    "kernel_output": out.tolist(),
    "reference_output": ref_out.tolist(),
    "ref_exact64_index": ref_idx,
    "exact64_dists": d.tolist(),
    "exact_dist_spread": float(d.max() - d.min()),
    "relative_l2": float(rel),
    "tolerance": 0.1,
    "kernel_matches_reference_row": bool(np.allclose(out.numpy(), ref_out)),
    "passes_contract": bool(rel <= 0.1),
}))