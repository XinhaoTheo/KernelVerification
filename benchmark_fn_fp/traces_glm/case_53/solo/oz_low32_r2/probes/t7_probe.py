import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_r")
import kernel

query, candidates, embeddings = kernel.make_inputs("cuda")

# reference: float64 exact distances, tie-break smallest index
q64 = query.double().cpu().numpy()
c64 = candidates.double().cpu().numpy()
d = ((c64 - q64[None, :])**2).sum(axis=1)
ref_idx = int(np.argmin(d))  # argmin picks first occurrence = smallest index on ties
ref = embeddings[ref_idx].float()

out = kernel.run(query, candidates, embeddings)
# kernel winner index: replicate quantized routing in numpy
q8 = np.floor(q64*8.0+0.5)*0.125
c8 = np.floor(c64*8.0+0.5)*0.125
dq = ((c8 - q8[None, :])**2).astype(np.float32).sum(axis=1, dtype=np.float32)
kernel_idx_sim = int(np.argmin(dq))

# also extract actual kernel winner: compare output to each embedding row
diffs = [float((embeddings[i].float()-out).norm().item()) for i in range(16)]
kernel_idx = int(np.argmin(diffs))

rel = float((out - ref.cuda()).norm().item()) / max(float(ref.norm().item()), 1e-12)
print(json.dumps({
  "ref_idx": ref_idx, "kernel_idx": kernel_idx, "kernel_idx_sim": kernel_idx_sim,
  "rel_l2": rel, "pass": rel <= 0.1,
  "ref_dist_min": float(d.min()), "ref_dist_sorted_head": [float(x) for x in np.sort(d)[:4]],
  "quant_dist_sorted_head": [float(x) for x in np.sort(dq)[:4]],
  "finite": bool(torch.isfinite(out).all().item()),
}))