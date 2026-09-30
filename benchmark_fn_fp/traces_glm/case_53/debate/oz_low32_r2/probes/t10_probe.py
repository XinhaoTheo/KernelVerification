
import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_r")
import kernel

query, candidates, embeddings = kernel.make_inputs()
out = kernel.run(query, candidates, embeddings)

# reference: float64 squared distances
q64 = query.numpy().astype(np.float64)
c64 = candidates.numpy().astype(np.float64)
e64 = embeddings.numpy().astype(np.float64)
d = ((c64 - q64[None, :]) ** 2).sum(axis=1)
ref_idx = int(np.argmin(d))
ref = e64[ref_idx]

# emulate kernel route to get winner index (sanity check)
def q(x): return np.floor(x.astype(np.float64) * 8.0 + 0.5) * 0.125
dq = ((q(c64) - q(q64)[None, :]) ** 2).sum(axis=1)
emu_idx = int(np.argmin(dq))

# infer kernel winner: which embedding row matches output
diffs = np.abs(e64 - out.cpu().numpy()[None, :]).sum(axis=1)
kernel_idx = int(np.argmin(diffs))

rel = np.linalg.norm(out.cpu().numpy() - ref) / max(np.linalg.norm(ref), 1e-12)
d_sorted = np.sort(d)
print(json.dumps({
    "reference_winner": ref_idx,
    "kernel_winner_inferred": kernel_idx,
    "emulated_winner": emu_idx,
    "winner_mismatch": bool(kernel_idx != ref_idx),
    "relative_l2_error": float(rel),
    "tolerance": 0.1,
    "contract_pass": bool(rel <= 0.1),
    "true_dist_min": float(d_sorted[0]),
    "true_dist_second": float(d_sorted[1]),
    "true_dist_gap": float(d_sorted[1] - d_sorted[0]),
    "true_dist_max": float(d_sorted[-1]),
    "output_finite": bool(np.isfinite(out.cpu().numpy()).all()),
}))
