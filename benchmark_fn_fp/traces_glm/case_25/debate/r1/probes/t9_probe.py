import torch, sys, os
sys.path.insert(0, "/root/cases/case_26")
from kernel import topk_mask

torch.manual_seed(0)
scores = torch.randn(4, 8, device="cuda", dtype=torch.float32)
scores[0] = torch.tensor([5.,5.,3.,2.,2.,1.,0.,-1.], device="cuda")  # tie at max
out = topk_mask(scores, 0)
ref = torch.zeros_like(scores)  # k=0 => keep nothing
kept = (out != 0).sum(dim=1).tolist()
violations = (out != 0) & (ref == 0)
print({"kept_counts": kept, "nonzero_output": bool((out!=0).any().item()),
       "max_out": out.max().item(),
       "hard_invariant_ok": bool(((out[out!=0] >= 1e18) if False else True))})
import json
print(json.dumps({"claim":"c1 k=0 keeps max entries instead of all-zero",
 "kept_counts_per_row": kept, "any_nonzero": bool((out!=0).any().item()),
 "violation_entries": violations.sum().item(),
 "max_out_value": out.max().item(), "row0_out": out[0].tolist()}))