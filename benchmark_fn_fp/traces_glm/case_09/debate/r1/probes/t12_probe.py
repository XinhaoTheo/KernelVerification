import torch, json, sys
sys.path.insert(0, "/root/cases/case_09")
from kernel import sorted_topk_indices

res = {}
dev = "cuda"
for N in [2, 4, 8, 16, 64]:
    B = 8
    rows = []
    # all-equal row
    rows.append(torch.full((N,), 0.5, device=dev))
    # duplicated max at various pairs
    for (i, j) in [(0, N // 2), (1, N - 2), (N // 2, N - 1)]:
        r = torch.randn(N, device=dev)
        m = r.max()
        r[i] = m; r[j] = m
        rows.append(r)
    # max repeated 3 times
    r = torch.randn(N, device=dev)
    m = r.max()
    for t in [0, N // 3, N - 1]:
        r[t] = m
    rows.append(r)
    scores = torch.stack(rows)
    got = sorted_topk_indices(scores, 1).squeeze(1)
    want = scores.argmax(1)
    mism = (got != want).nonzero().flatten().tolist()
    res[f"N={N}"] = {"mismatch_rows": mism, "got": got.tolist(), "want": want.tolist(),
                     "n_rows": scores.shape[0]}
print(json.dumps({"claim": "c1 tie-break", "metric": "exact index equality vs torch.argmax (lowest tied index)", "results": res}))