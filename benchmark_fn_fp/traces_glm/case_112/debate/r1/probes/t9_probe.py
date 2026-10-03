import sys, json, torch
sys.path.insert(0, "/root/cases/case_112")
import kernel as k

torch.manual_seed(0)
gen = torch.Generator().manual_seed(1)
L, K = 97, 32
u = (2*torch.rand((2, L, 2, 33), generator=gen)-1).float().cuda()
decay = (0.90 + 0.06*torch.rand((2, L, 2), generator=gen)).float().cuda()
init = (2*torch.rand((2, 2, 33), generator=gen)-1).float().cuda()
seq = torch.zeros((2, L), dtype=torch.int32)
# boundary strictly inside chunk 1 (tokens 32..63): jump at token 40, non-consecutive label jump
seq[:, 40:] += 1
seq[:, 80:] += 3
seq = seq.cuda()

out, final = k.run(u, decay, seq, init, K)
ref_out, ref_final = k.reference(u, decay, seq, init, K)
ref_out32, ref_final32 = ref_out.float(), ref_final.float()

err = (out - ref_out32).abs()
tol = 0.002 + 1e-4 * ref_out32.abs()
mask = err > tol
# error specifically in tokens 40..63 (chunk 1 after boundary)
chunk1 = torch.zeros(L, dtype=torch.bool, device=err.device); chunk1[40:64] = True
mask1 = mask[:, chunk1].any(dim=tuple(range(2, mask[:, chunk1].ndim))) if False else mask[:, chunk1].all(dim=0) if mask.ndim>2 else None
# simpler: per-token max err
tok_err = err.amax(dim=(0,2,3))
print(json.dumps({
  "metric": "max_abs_err vs FP64 reference, tolerance breach count",
  "L": L, "K": K, "boundary_tokens": [40, 80],
  "max_abs_err": err.max().item(),
  "breach_count": int(mask.sum()),
  "breach_tokens_in_chunk1_after_boundary": sorted(set((mask.nonzero()[:,1]).tolist()))[:20],
  "tok_err_max_chunk1_after_boundary": tok_err[40:64].max().item(),
  "tok_err_max_overall": tok_err.max().item(),
  "tol": 0.002,
}))