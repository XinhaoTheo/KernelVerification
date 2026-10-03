import torch, json, sys
sys.path.insert(0, "/root/cases/case_113")
import kernel as K

torch.manual_seed(1)
device = "cuda"
B, L, H, D = 1, 96, 2, 33
chunk = 32
u = (2*torch.rand((B,L,H,D), device=device)-1)
decay = (0.90 + 0.06*torch.rand((B,L,H), device=device))
init = (2*torch.rand((B,H,D), device=device)-1)
# break exactly at token 32 = chunk 1's first token
seq = torch.zeros((B,L), dtype=torch.int32, device=device)
seq[:, 32:] += 1
out, fin = K.run(u, decay, seq, init, chunk)
ref_out, ref_fin = K.reference(u, decay, seq, init, chunk)
tol = 0.002 + 0.0001*ref_out.abs()
err = (out - ref_out).abs()
bad = err > tol
tolf = 0.002 + 0.0001*ref_fin.abs()
badf = (fin - ref_fin).abs() > tolf
print(json.dumps({
  "max_abs_err": float(err.max()),
  "tol": float(tol.max()),
  "violations": int(bad.sum()),
  "total": out.numel(),
  "final_max_err": float((fin - ref_fin).abs().max()),
  "final_violations": int(badf.sum()),
  "break_token": 32,
}))
