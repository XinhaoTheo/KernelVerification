import torch, sys
sys.path.insert(0, '/root/cases/case_13')
from kernel import gqa_gather

kv = torch.arange(2*4, dtype=torch.float32, device='cuda').reshape(2,4)
out = gqa_gather(kv, 4)
ref = torch.repeat_interleave(kv, 2, dim=0)
print(out)
print(ref)
mismatch = (out != ref).sum().item()
print({"mismatched_elements": mismatch, "total": ref.numel(), "num_kv_heads": 2, "num_q_heads": 4})
assert mismatch > 0
