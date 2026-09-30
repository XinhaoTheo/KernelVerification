import numpy as np, json, sys
sys.path.insert(0, "/root/numerical_cases/case_q")
from kernel import make_inputs_numpy

query, candidates, embeddings = make_inputs_numpy()

# exact float64 route
q64 = query.astype(np.float64)
d_exact = ((candidates.astype(np.float64) - q64) ** 2).sum(axis=1)
ref_idx = int(np.argmin(d_exact))

# quantized float32 route mirroring kernel lines 11-20
def quant(x):
    return np.floor(x.astype(np.float32) * np.float32(8.0) + np.float32(0.5)).astype(np.float32) * np.float32(0.125)
q_q = quant(query)
c_q = quant(candidates)
dist = np.zeros(16, dtype=np.float32)
for i in range(16):
    acc = np.float32(0.0)
    for j in range(32):
        delta = np.float32(c_q[i, j] - q_q[j])
        acc = np.float32(acc + np.float32(delta * delta))
    dist[i] = acc
minimum = dist.min()
tied = np.where(dist == minimum)[0]
qwin = int(tied.min())  # smallest-index tie rule

rel = float(np.linalg.norm(embeddings[qwin].astype(np.float64) - embeddings[ref_idx].astype(np.float64)) /
            max(np.linalg.norm(embeddings[ref_idx].astype(np.float64)), 1e-12))

print(json.dumps({
    "metric": "quantized-space tie presence and smallest-index resolution vs exact float64 winner",
    "exact64_ref_index": ref_idx,
    "quantized_winner_index": qwin,
    "quantized_min_distance": float(minimum),
    "n_tied_at_min": int(len(tied)),
    "tied_indices": tied.tolist(),
    "quantized_dists": dist.tolist(),
    "exact64_dists": d_exact.tolist(),
    "tie_changes_index_vs_exact": bool(qwin != ref_idx),
    "relative_l2_if_wrong": rel,
    "tolerance": 0.1,
}))