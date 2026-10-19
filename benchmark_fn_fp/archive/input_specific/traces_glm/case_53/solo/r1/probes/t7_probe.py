import numpy as np, torch, json, sys
sys.path.insert(0, "/root/numerical_cases/case_r")
from kernel import run, make_inputs_numpy

query, candidates, embeddings = make_inputs_numpy()
q = query.astype(np.float64)
d = ((candidates.astype(np.float64) - q)**2).sum(axis=1)
ref_idx = int(np.lexsort((np.arange(len(d)), d))[0])
ref = embeddings[ref_idx]

out = run(torch.from_numpy(query).cuda(), torch.from_numpy(candidates).cuda(), torch.from_numpy(embeddings).cuda()).cpu().numpy()
kernel_idx = int(np.argmin(np.linalg.norm(embeddings - out[None,:], axis=1)))
rel = np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-12)
# also compute quantized winner as kernel would
qq = np.floor(q*8+0.5)/8
cq = np.floor(candidates.astype(np.float64)*8+0.5)/8
dq = ((cq-qq)**2).sum(axis=1)
q_idx = int(np.argmin(dq))
print(json.dumps({"ref_idx": ref_idx, "quant_idx": q_idx, "kernel_idx": kernel_idx, "rel_err": float(rel), "d_sorted_head": sorted(d)[:4].tolist(), "dq_sorted_head": sorted(dq)[:4].tolist(), "output": out.tolist(), "ref": ref.tolist()}))
