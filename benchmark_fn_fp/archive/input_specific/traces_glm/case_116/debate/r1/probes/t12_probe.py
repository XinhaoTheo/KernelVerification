import sys, json, numpy as np, torch
sys.path.insert(0, "/root/numerical_cases/case_r")
from kernel import make_inputs, run

query, candidates, embeddings = make_inputs()
q64 = query.detach().cpu().numpy().astype(np.float64)
c64 = candidates.detach().cpu().numpy().astype(np.float64)
e = embeddings.detach().cpu().numpy().astype(np.float64)
d64 = ((c64 - q64[None, :])**2).sum(axis=1)
ref_idx = int(np.argmin(d64))

# emulate kernel quantized route in float32
qf = query.detach().cpu().numpy().astype(np.float32)
cf = candidates.detach().cpu().numpy().astype(np.float32)
qq = np.floor(qf*8.0+0.5)*0.125
cq = np.floor(cf*8.0+0.5)*0.125
dq = ((cq.astype(np.float32)-qq[None,:].astype(np.float32))**2).astype(np.float32).sum(axis=1, dtype=np.float32)
quant_idx = int(np.argmin(dq.numpy() if hasattr(dq,'numpy') else dq))

out = run(query, candidates, embeddings).detach().cpu().numpy().astype(np.float64)
ref = e[ref_idx]
rel = float(np.linalg.norm(out-ref)/max(np.linalg.norm(ref),1e-12))

# also actual kernel index inferred from output row match
kernel_row = int(np.argmin([np.linalg.norm(out - e[i]) for i in range(16)]))
print(json.dumps({
  "claim":"c1","metric":"quantized vs float64 argmin and relative L2 of output",
  "ref_idx_f64":ref_idx,"quant_idx_emulated":quant_idx,"kernel_row_inferred":kernel_row,
  "indices_match": bool(ref_idx==quant_idx),
  "d64_min":float(d64.min()),"d64_max":float(d64.max()),"d64_spread":float(d64.max()-d64.min()),
  "rel_l2_kernel_output":rel,"tolerance":0.1,"output_finite":bool(np.isfinite(out).all()),
  "rel_l2_if_index_flip": float(np.linalg.norm(e[quant_idx]-e[ref_idx])/max(np.linalg.norm(e[ref_idx]),1e-12)) if quant_idx!=ref_idx else 0.0
}))
