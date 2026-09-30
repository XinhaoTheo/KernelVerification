import torch, json, sys
sys.path.insert(0, "/root/cases/case_10")
from kernel import sorted_topk_indices

dev = "cuda"
N, k = 16, 8
rows = []
# Row 0: ties straddling k cutoff (idxs 1,3,5,8 tied at 1.0)
rows.append(torch.tensor([5.0,1.0,4.0,1.0,3.0,1.0,2.0,1.0, 1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], device=dev))
# Row 1: clustered ties at cutoff (idxs 6,7,8,9 tied at 0.5)
rows.append(torch.tensor([2.0,2.0,1.0,1.0,1.0,1.0,0.5,0.5, 0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25], device=dev))
# Row 2: two-way tie at cutoff exactly (idxs 6,7 tied at 3.0)
rows.append(torch.tensor([9.0,8.0,7.0,6.0,5.0,4.0,3.0,3.0, 2.0,1.0,0.0,-1.0,-2.0,-3.0,-4.0,-5.0], device=dev))
scores = torch.stack(rows)

out = sorted_topk_indices(scores, k)
torch.cuda.synchronize()
ref = torch.argsort(-scores.float(), dim=-1, stable=True)[:, :k]

res = {
  "kernel_selected": out.tolist(),
  "stable_desc_ref": ref.tolist(),
  "kernel_matches_stable_ref": bool(torch.equal(out, ref)),
  "row0_kernel": out[0].tolist(), "row0_ref": ref[0].tolist(),
  "row1_kernel": out[1].tolist(), "row1_ref": ref[1].tolist(),
  "row2_kernel": out[2].tolist(), "row2_ref": ref[2].tolist(),
}
print(json.dumps(res))