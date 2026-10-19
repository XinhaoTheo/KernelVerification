import torch, json, sys
sys.path.insert(0, "/root/cases/case_10")
from kernel import sorted_topk_indices

torch.manual_seed(0)
dev = "cuda"
N, k = 16, 8
scores = torch.randn(4, N, device=dev, dtype=torch.float32)
scores[0, 5] = float("nan")
scores[1, 0] = float("nan"); scores[1, 7] = float("nan")
scores[2, 15] = float("nan")  # NaN at last index
# row 3: all normal control

out = sorted_topk_indices(scores, k)
torch.cuda.synchronize()
ref = torch.topk(scores, k, dim=-1).indices

sel_vals = torch.gather(scores, 1, out.to(dev))
res = {
  "kernel_selected": out.tolist(),
  "torch_topk_selected": ref.tolist(),
  "kernel_selected_values": sel_vals.tolist(),
  "nan_in_kernel_selection": bool(torch.isnan(sel_vals).any().item()),
  "nan_positions_in_kernel_output": [[out[b].tolist().index(i) if i in out[b].tolist() else None for i in torch.nonzero(torch.isnan(scores[b])).flatten().tolist()] for b in range(4)],
  "nan_positions_in_topk_output": [[ref[b].tolist().index(i) if i in ref[b].tolist() else None for i in torch.nonzero(torch.isnan(scores[b])).flatten().tolist()] for b in range(4)],
  "row3_control_match": bool(torch.equal(out[3], ref[3])),
}
print(json.dumps(res))