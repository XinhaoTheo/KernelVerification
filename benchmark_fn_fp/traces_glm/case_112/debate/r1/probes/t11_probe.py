import sys, json, torch
sys.path.insert(0, "/root/cases/case_112")
import kernel as k

gen = torch.Generator().manual_seed(3)
L = 97
u = (2*torch.rand((1, L, 2, 33), generator=gen)-1).float().cuda()
decay = (0.90 + 0.06*torch.rand((1, L, 2), generator=gen)).float().cuda()
init = (2*torch.rand((1, 2, 33), generator=gen)-1).float().cuda()
seq = torch.zeros((1, L), dtype=torch.int32)
seq[:, 40:] += 1
seq[:, 80:] += 3
seq = seq.cuda()

results = {}
for K in (16, 32, 64):
    out, final = k.run(u, decay, seq, init, K)
    ref_out, ref_final = k.reference(u, decay, seq, init, K)
    r = ref_out.float()
    err = (out - r).abs()
    tol = 0.002 + 1e-4 * r.abs()
    ferr = (final - ref_final.float()).abs()
    ftol = 0.002 + 1e-4 * ref_final.float().abs()
    results[K] = {
        "out_max_abs_err": err.max().item(),
        "out_breach_count": int((err > tol).sum()),
        "final_max_abs_err": ferr.max().item(),
        "final_breach_count": int((ferr > ftol).sum()),
    }
# boundary at chunk edge for K=32: jump exactly at token 32
seq2 = torch.zeros((1, L), dtype=torch.int32).cuda()
seq2[:, 32:] += 1
res_edge = {}
for K in (16, 32, 64):
    out, final = k.run(u, decay, seq2, init, K)
    ref_out, ref_final = k.reference(u, decay, seq2, init, K)
    r = ref_out.float()
    err = (out - r).abs()
    tol = 0.002 + 1e-4 * r.abs()
    res_edge[K] = {"out_max_abs_err": err.max().item(),
                   "out_breach_count": int((err > tol).sum())}
print(json.dumps({
  "metric": "chunk_size invariance: breach counts for chunk sizes 16/32/64 with mid-chunk boundary at 40/80 and edge-aligned boundary at 32",
  "mid_boundary": results,
  "edge_boundary_at_32": res_edge,
  "tol": 0.002,
}))