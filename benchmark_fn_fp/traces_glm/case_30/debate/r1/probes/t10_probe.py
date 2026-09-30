import torch, json
import sys
sys.path.insert(0, "/root/cases/case_30")
from kernel import route_top1

dev = "cuda"
results = {}

# Case A: all-equal logits row (full tie) - fp32
E = 8; T = 4
lg = torch.full((T, E), 0.5, device=dev, dtype=torch.float32)
idx = route_top1(lg)
expect = torch.zeros(T, dtype=torch.long, device=dev)
results["all_equal_fp32_kernel"] = idx.tolist()
results["all_equal_fp32_expected"] = expect.tolist()
results["all_equal_fp32_mismatch_rows"] = int((idx.long() != expect).sum().item())

# Case B: partial tie - max shared by experts 2 and 5, lowest should win
lg2 = torch.randn(T, E, device=dev, dtype=torch.float32)
lg2[:, :] = torch.linspace(-1, 1, E, device=dev).unsqueeze(0)
lg2[:, 2] = 1.0; lg2[:, 5] = 1.0  # tied max at indices 2 and 5
idx2 = route_top1(lg2)
results["partial_tie_fp32_kernel"] = idx2.tolist()
results["partial_tie_fp32_expected"] = [2]*T
results["partial_tie_fp32_mismatch_rows"] = int((idx2.long() != 2).sum().item())

# Case C: bf16 quantization-induced ties (in-scope primary distribution)
lg3 = torch.randn(256, E, device=dev, dtype=torch.float32).to(torch.bfloat16)
idx3 = route_top1(lg3)
# reference: lowest tied index wins
ref3 = torch.argmax(lg3.float(), dim=1)  # torch.argmax returns first occurrence of max
# verify ref picks lowest tied index by explicit check
vals = lg3.float()
mx = vals.max(dim=1, keepdim=True).values
lowest = ((vals == mx).float() * torch.arange(E, device=dev).unsqueeze(0)).masked_fill(vals != mx, float("inf")).min(dim=1).values.long()
results["bf16_ties_rows_with_tied_max"] = int(((vals == mx).sum(dim=1) > 1).sum().item())
results["bf16_kernel_mismatch_vs_lowest"] = int((idx3.long() != lowest).sum().item())
results["bf16_kernel_mismatch_vs_torchargmax"] = int((idx3.long() != ref3).sum().item())
results["bf16_rows"] = int(lg3.shape[0])

# Case D: no ties (distinct logits) - sanity that argmax value itself is right
lg4 = torch.randn(T, E, device=dev, dtype=torch.float32)
idx4 = route_top1(lg4)
ref4 = lg4.argmax(dim=1)
results["distinct_fp32_correct_rows"] = int((idx4.long() == ref4).sum().item())

print(json.dumps(results))
