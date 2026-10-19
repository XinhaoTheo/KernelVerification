import torch, json, sys
sys.path.insert(0, "/root/cases/case_09")
from kernel import sorted_topk_indices

res = {}
dev = "cuda"
for N in [2, 4, 8, 16, 64]:
    B = 16
    # reversed-range rows: max at 0, min at N-1 (descending unique values)
    desc = torch.arange(N, 0, -1, device=dev, dtype=torch.float32).unsqueeze(0).expand(B // 2, N)
    # ascending-range: max at N-1, min at 0
    asc = torch.arange(1, N + 1, device=dev, dtype=torch.float32).unsqueeze(0).expand(B // 2, N)
    rand = torch.randn(B, N, device=dev)
    scores = torch.cat([desc, asc, rand])[:B]
    got = sorted_topk_indices(scores, 1).squeeze(1)
    want = scores.argmax(1)
    mism = (got != want).nonzero().flatten().tolist()
    res[f"N={N}"] = {"mismatch_rows": mism, "n_rows": scores.shape[0],
                     "desc_row_got": got[0].item(), "desc_row_want": want[0].item(),
                     "asc_row_got": got[B // 2].item(), "asc_row_want": want[B // 2].item()}
print(json.dumps({"claim": "c2 order direction", "metric": "exact index equality vs torch.argmax on unique-max rows", "results": res}))