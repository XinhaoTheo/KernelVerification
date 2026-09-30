import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_r")
from kernel import make_inputs

query, candidates, embeddings = make_inputs()
q64 = query.detach().cpu().numpy().astype(np.float64)
c64 = candidates.detach().cpu().numpy().astype(np.float64)
e = embeddings.detach().cpu().numpy().astype(np.float64)
d64 = ((c64 - q64[None, :])**2).sum(axis=1)
ref_idx = int(np.argmin(d64))

qf = query.detach().cpu().numpy().astype(np.float32)
cf = candidates.detach().cpu().numpy().astype(np.float32)
qq = np.floor(qf*8.0+0.5)*0.125
cq = np.floor(cf*8.0+0.5)*0.125
dq = ((cq.astype(np.float32)-qq[None,:].astype(np.float32))**2).astype(np.float32).sum(axis=1, dtype=np.float32)
dq = np.asarray(dq, dtype=np.float32)
qmin = dq.min()
tied = [int(i) for i in range(16) if dq[i]==qmin]
qarg = tied[0]  # smallest index tie-break on quantized distances

rel = float(np.linalg.norm(e[qarg]-e[ref_idx])/max(np.linalg.norm(e[ref_idx]),1e-12))
print(json.dumps({
  "claim":"c2","metric":"exact ties at quantized minimum and resulting embedding error",
  "quant_min_dist":float(qmin),
  "tied_indices_at_quant_min":tied,
  "num_tied":len(tied),
  "quant_smallest_index":qarg,
  "ref_idx_f64":ref_idx,
  "float64_min_dist":float(d64.min()),
  "ref_ties_f64": [int(i) for i in range(16) if d64[i]==d64.min()],
  "rel_l2_between_selections":rel,"tolerance":0.1,
  "tie_changes_selection": bool(qarg!=ref_idx)
}))
