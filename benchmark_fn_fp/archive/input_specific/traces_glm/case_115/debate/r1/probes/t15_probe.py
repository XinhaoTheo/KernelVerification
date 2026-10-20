import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_q")
from kernel import run, make_inputs

query, candidates, embeddings = make_inputs()
out = run(query, candidates, embeddings)

# exact float64 reference (host copies)
q = query.cpu().to(torch.float64).numpy()
c = candidates.cpu().to(torch.float64).numpy()
e = embeddings.cpu().numpy().astype(np.float64)
d = ((c - q) ** 2).sum(axis=1)
ref_idx = int(np.argmin(d))
ref_out = e[ref_idx]
out_np = out.cpu().numpy().astype(np.float64)
rel = float(np.linalg.norm(out_np - ref_out) / max(np.linalg.norm(ref_out), 1e-12))

# also identify which embedding row the kernel actually returned
row_errs = [float(np.linalg.norm(out_np - e[i])) for i in range(16)]
kernel_idx = int(np.argmin(row_errs))

print(json.dumps({
    "metric": "kernel output vs exact float64 reference on fixed seed-840104 workload",
    "ref_exact64_index": ref_idx,
    "kernel_returned_row_index": kernel_idx,
    "relative_l2": rel,
    "tolerance": 0.1,
    "winner_match": bool(kernel_idx == ref_idx),
    "passes_contract": bool(rel <= 0.1),
    "exact64_dists": d.tolist(),
}))