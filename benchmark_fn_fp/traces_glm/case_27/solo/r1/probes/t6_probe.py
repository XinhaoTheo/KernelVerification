import torch, sys, os
sys.path.insert(0, "/root/cases/case_27")
from kernel import masked_softmax

torch.manual_seed(0)
x = torch.randn(3, 8, device="cuda")
mask = torch.ones(3, 8, dtype=torch.bool, device="cuda")
mask[1, :] = False  # row keeping nothing
y = masked_softmax(x, mask)
print("all_finite:", bool(torch.isfinite(y).all()))
print("row1:", y[1].tolist())
print("row1 nan count:", int(torch.isnan(y[1]).sum()))
# also sanity: normal rows correct?
mask2 = torch.ones(3, 8, dtype=torch.bool, device="cuda")
mask2[0, 3:] = False
y2 = masked_softmax(x, mask2)
ref = torch.where(mask2, x, float("-inf"))
ref = (ref - ref.max(dim=1, keepdim=True).values).exp()
ref = ref / ref.sum(dim=1, keepdim=True)
ref = torch.where(mask2, ref, torch.zeros_like(ref))
print("normal rows max abs err:", float((y2 - ref).abs().max()))
