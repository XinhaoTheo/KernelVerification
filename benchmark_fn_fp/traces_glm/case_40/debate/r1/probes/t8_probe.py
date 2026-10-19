import torch, json, sys
sys.path.insert(0, "/root/cases/case_113")
import kernel as K

torch.manual_seed(0)
device = "cuda"
B, L, H, D = 1, 100, 2, 33
chunk = 32
u = (2*torch.rand((B,L,H,D), device=device)-1)
decay = (0.90 + 0.06*torch.rand((B,L,H), device=device))
init = (2*torch.rand((B,H,D), device=device)-1)
# break strictly inside chunk 1 (tokens 32..63), e.g. token 50
seq = torch.zeros((B,L), dtype=torch.int32, device=device)
seq[:, 50:] += 1
out, fin = K.run(u, decay, seq, init, chunk)
ref_out, ref_fin = K.reference(u, decay, seq, init, chunk)
tol = 0.002 + 0.0001*ref_out.abs()
err = (out - ref_out).abs()
bad = err > tol
print(json.dumps({
  "max_abs_err": float(err.max()),
  "tol": float(tol.max()),
  "violations": int(bad.sum()),
  "total": out.numel(),
  "final_max_err": float((fin - ref_fin).abs().max()),
  "violation_tokens": sorted(set(torch.nonzero(bad)[0].tolist()))[:10] if bad.any() else [],
}))
