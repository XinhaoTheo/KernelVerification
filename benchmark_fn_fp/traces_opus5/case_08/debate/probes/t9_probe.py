
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k9", "/root/cases/case_09/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

rows = []
cases = []
# N=4: duplicate max at (i,j) for all i<j, rest lower distinct
for N in [2, 4, 8, 16]:
    for i in range(N):
        for j in range(i+1, N):
            r = torch.full((N,), -1.0)
            r[i] = 1.0; r[j] = 1.0
            cases.append((N, i, j, r))
summary = {}
for N in [2, 4, 8, 16]:
    sub = [(i, j, r) for (n, i, j, r) in cases if n == N]
    scores = torch.stack([r for (_, _, r) in sub]).cuda()
    top1 = m.sorted_topk_indices(scores, 1).squeeze(1).cpu()
    lo = torch.tensor([i for (i, _, _) in sub])
    hi = torch.tensor([j for (_, j, _) in sub])
    got_low = int((top1 == lo).sum()); got_high = int((top1 == hi).sum())
    other = int(((top1 != lo) & (top1 != hi)).sum())
    summary[str(N)] = {"pairs": len(sub), "returned_lower_index": got_low,
                       "returned_higher_index": got_high, "returned_neither": other,
                       "sample": [[int(lo[t]), int(hi[t]), int(top1[t])] for t in range(min(6, len(sub)))]}
print(json.dumps({"tie_probe": summary,
                  "metric": "returned k=1 index vs lowest tied max index (contract: LOWER wins)"}))
