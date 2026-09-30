import torch, json, math, sys
sys.path.insert(0, "/root/cases/case_10")
from kernel import sorted_topk_indices

torch.manual_seed(0)
dev = "cuda"
N = 16  # power of two
k = 8
# Row 0: one NaN among normal scores; Row 1: NaN at low index; Row 2: all normal control
scores = torch.randn(3, N, device=dev, dtype=torch.float32)
scores[0, 5] = float("nan")
scores[1, 0] = float("nan")
scores[1, 7] = float("nan")

out = sorted_topk_indices(scores, k)
torch.cuda.synchronize()
ref = torch.topk(scores, k, dim=-1).indices

# also inspect the sorted values returned implicitly: reconstruct values of selected idxs
sel_vals = torch.gather(scores, 1, out.to(scores.device))
# check whether any NaN index is selected and where
nan_pos_kernel = []
for b in range(3):
    row_nan_mask = torch.isnan(scores[b])
    nan_ids = torch.nonzero(row_nan_mask).flatten().tolist()
    sel = out[b].tolist()
    nan_pos_kernel.append([ (sel.index(i) if i in sel else None) for i in nan_ids ])
nan_pos_ref = []
for b in range(3):
    row_nan_mask = torch.isnan(scores[b])
    nan_ids = torch.nonzero(row_nan_mask).flatten().tolist()
    sel = ref[b].tolist()
    nan_pos_ref.append([ (sel.index(i) if i in sel else None) for i in nan_ids ])

non_nan_agree = []
for b in range(3):
    # compare selected indices excluding NaN entries
    m = ~torch.isnan(scores[b])
    non_nan_agree.append(bool(torch.equal(out[b][torch.gather(m.expand(k),0,torch.arange(k,device=dev))[~torch.isnan(torch.gather(scores[b],0,out[b]).cpu())] ] if False else torch.tensor(True, device=dev))))

# simpler: check sorted value sequence of selected (NaN positions)
sel_vals_list = sel_vals.tolist()
res = {
  "kernel_selected": out.tolist(),
  "torch_topk_selected": ref.tolist(),
  "kernel_nan_selected_positions": nan_pos_kernel,
  "topk_nan_selected_positions": nan_pos_ref,
  "kernel_selected_values": sel_vals_list,
  "nan_in_kernel_selection": bool(torch.isnan(sel_vals).any().item()),
}
print(json.dumps(res))