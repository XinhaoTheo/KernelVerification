import torch, json, sys
sys.path.insert(0, "/root/cases/case_10")
from kernel import sorted_topk_indices

torch.manual_seed(0)
dev = "cuda"
N, k = 16, 8
rows = []
# Row 0: ties straddling the k cutoff (values equal at boundary), lower index should be kept
r = torch.tensor([5.0,1.0,4.0,1.0,3.0,1.0,2.0,1.0, 1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], device=dev)
rows.append(r)
# Row 1: many duplicated values, heavier tie clustering
r = torch.tensor([2.0,2.0,1.0,1.0,1.0,1.0,0.5,0.5, 0.5,0.5,0.25,0.25,0.25,0.25,0.25,0.25], device=dev)
rows.append(r)
# Row 2: two-way tie at cutoff exactly
r = torch.tensor([9.0,8.0,7.0,6.0,5.0,4.0,3.0,3.0, 2.0,1.0,0.0,-1.0,-2.0,-3.0,-4.0,-5.0], device=dev)
rows.append(r)
scores = torch.stack(rows)

out = sorted_topk_indices(scores, k)
torch.cuda.synchronize()
# Reference: descending sort with stable lower-index tie order
ref = torch.argsort(-scores.float(), dim=-1, stable=True)[:, :k]
# also raw torch.topk for comparison
tk = torch.topk(scores, k, dim=-1).indices

res = {
  "kernel_selected": out.tolist(),
  "stable_argsort_desc_ref": ref.tolist(),
  "torch_topk_selected": tk.tolist(),
  "kernel_matches_stable_ref": bool(torch.equal(out, ref)),
  "kernel_matches_topk": bool(torch.equal(out, tk)),
}
print(json.dumps(res))