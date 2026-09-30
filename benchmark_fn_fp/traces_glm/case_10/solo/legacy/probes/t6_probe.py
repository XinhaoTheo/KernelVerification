import sys, json, torch
sys.path.insert(0, "/root/cases/case_10")
from kernel import sorted_topk_indices

results = {}
torch.manual_seed(0)
dev = "cuda"

# 1) all-equal scores: ties everywhere; contract wants indices [0..k-1]
N, B, k = 16, 4, 4
s = torch.zeros(B, N, device=dev)
idx = sorted_topk_indices(s, k)
results["all_equal"] = idx.tolist()
ref = torch.arange(N, device=dev).unsqueeze(0).expand(B, N)[:, :k].long()
results["all_equal_ref"] = ref.tolist()
results["all_equal_match"] = bool((idx == ref).all())

# 2) half ties: scores 0,0,0,0,1,2,3,4,... distinct top half
N2 = 16
s2 = torch.zeros(B, N2, device=dev)
s2[:, 8:] = torch.arange(1, 9, device=dev, dtype=torch.float32).unsqueeze(0)
# k=4 -> distinct 5,6,7,8 values: indices 12,13,14,15
idx2 = sorted_topk_indices(s2, 4)
results["distinct_top"] = idx2.tolist()

# 3) ties at the cutoff: values 8,7,6,5 then five 4s, then lower
s3 = torch.tensor([[8,7,6,5,4,4,4,4,4,3,2,1,0,0,0,0]], dtype=torch.float32, device=dev)
idx3 = sorted_topk_indices(s3, 7)  # 8,7,6,5 + two of the tied 4s -> indices 8 and 9? lower idx kept: 4,5
results["cutoff_ties"] = idx3.tolist()

# 4) random with quantized scores forcing many ties
s4 = (torch.randn(64, 16, device=dev) * 2).round().to(torch.float32)  # scores in small range -> many ties
idx4 = sorted_topk_indices(s4, 8)
ref4 = torch.argsort(-s4, dim=1, stable=True)[:, :8]
# compare selected values and whether each selected index is valid under stable ref set
results["rand_ties_match_stable_ref"] = bool((idx4 == ref4).all())
mism = (idx4 != ref4)
if mism.any():
    rows = mism.any(1).nonzero().flatten()[:5].tolist()
    examples = []
    for r in rows:
        tied = (s4[r].unsqueeze(0) == s4[r, idx4[r]].unsqueeze(1)).any(1)
        examples.append({"row": r, "got": idx4[r].tolist(), "ref": ref4[r].tolist(),
                          "scores": s4[r].tolist(),
                          "all_tied": bool(tied.all().item())})
    results["rand_tie_examples"] = examples

# count tie violations across many random quantized rows
viol = 0
for _ in range(20):
    s5 = (torch.randn(64, 32, device=dev) * 2).round()
    i5 = sorted_topk_indices(s5, 16)
    r5 = torch.argsort(-s5, dim=1, stable=True)[:, :16]
    viol += int((i5 != r5).sum().item())
results["total_violations_20x64x32"] = viol

print(json.dumps(results))