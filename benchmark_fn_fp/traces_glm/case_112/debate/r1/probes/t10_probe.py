import sys, json, torch
sys.path.insert(0, "/root/cases/case_112")
import kernel as k

gen = torch.Generator().manual_seed(2)
L, K = 97, 32
u = (2*torch.rand((1, L, 2, 33), generator=gen)-1).float().cuda()
decay = (0.90 + 0.06*torch.rand((1, L, 2), generator=gen)).float().cuda()
init = (2*torch.rand((1, 2, 33), generator=gen)-1).float().cuda()
seq = torch.zeros((1, L), dtype=torch.int32)
seq[:, 40:] += 1  # boundary mid-chunk (chunk 1), last chunk 2 (tokens 64..96) has no boundary
seq = seq.cuda()

out, final = k.run(u, decay, seq, init, K)
ref_out, ref_final = k.reference(u, decay, seq, init, K)
ref_final32 = ref_final.float()

ferr = (final - ref_final32).abs()
ftol = 0.002 + 1e-4 * ref_final32.abs()
print(json.dumps({
  "metric": "final_state abs error vs FP64 reference vs tolerance (0.002 + 1e-4*|ref|)",
  "L": L, "K": K, "boundary_at": 40, "last_chunk_boundary": False,
  "final_max_abs_err": ferr.max().item(),
  "final_breach_count": int((ferr > ftol).sum()),
  "final_numel": final.numel(),
  "tol_at_max": ftol.max().item(),
}))