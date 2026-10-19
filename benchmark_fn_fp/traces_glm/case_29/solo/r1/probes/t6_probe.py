import torch, sys
sys.path.insert(0, "/root/cases/case_30")
from kernel import route_top1

logits = torch.tensor([[3.0, 3.0, 3.0, 1.0],
                       [1.0, 3.0, 3.0, 0.0],
                       [2.0, 1.0, 0.5, 2.0]], device="cuda")
out = route_top1(logits)
print("kernel :", out.tolist())
print("expected (lowest tied index): [0, 1, 0]")
