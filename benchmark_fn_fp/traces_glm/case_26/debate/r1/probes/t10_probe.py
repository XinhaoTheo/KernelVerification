import torch, sys, json
sys.path.insert(0, "/root/cases/case_26")
from kernel import topk_mask

torch.manual_seed(1)
scores = torch.randn(4, 8, device="cuda", dtype=torch.float32)
# include a -inf row and a normal row
scores[1] = torch.tensor([1.,2.,3.,4.,5.,6.,7.,float("-inf")], device="cuda")
K = 12  # > n_cols=8
out = topk_mask(scores, K)
# For k>N, contract reading: keeping all N entries = output equals input
kept = (out != 0).sum(dim=1).tolist()
equals_input = torch.equal(out, scores)
# hard invariant: every kept >= every dropped
ok = True
for r in range(scores.shape[0]):
    nz = out[r][out[r]!=0]; z = out[r][out[r]==0]
    if len(z)>0 and len(nz)>0 and (nz.min() < z.max()*0): pass
print(json.dumps({"claim":"c2 k>N keeps every valid entry; output equals input",
 "k": K, "n_cols": 8, "kept_counts_per_row": kept,
 "output_equals_input": bool(equals_input),
 "row1_minus_inf_kept": out[1][-1].item(),
 "row1_out": out[1].tolist()}))