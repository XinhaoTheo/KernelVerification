import json, torch, sys
sys.path.insert(0, "/root/cases/case_26")
from kernel import topk_mask

torch.manual_seed(0)
results = []
worst = None
for trial, (N, K, dist) in enumerate([(256, 10, "randn"), (1024, 50, "randn"),
                                      (64, 32, "clustered"), (256, 10, "tied"),
                                      (256, 10, "cutoff_cluster")]):
    if dist == "randn":
        s = torch.randn(64, N, device="cuda")
    elif dist == "clustered":
        s = torch.randn(64, N, device="cuda") * 0.01
    elif dist == "tied":
        s = torch.randint(0, 5, (64, N), device="cuda").float()
    else:
        s = torch.randn(64, N, device="cuda")
        s[:, 5:40] = s[:, 5]  # many values equal/adjacent near cutoff
    out = topk_mask(s, K)
    torch.cuda.synchronize()
    inv_ok = True
    topk_ok = True
    kept_counts = []
    for r in range(s.shape[0]):
        row = s[r]
        o = out[r]
        kept = o != 0
        dropped = ~kept
        kept_counts.append(int(kept.sum()))
        if kept.sum() > 0 and dropped.sum() > 0:
            if o[kept].min() < row[dropped].max():
                inv_ok = False
        # all entries strictly greater than the k-th largest must be kept
        kth = torch.topk(row, K).values.min()
        if (o[row > kth] == 0).any():
            topk_ok = False
    key = (N, K, dist)
    results.append({"case": key, "invariant_kept_ge_dropped": inv_ok,
                    "all_above_kth_kept": topk_ok,
                    "kept_count_range": [min(kept_counts), max(kept_counts)]})
    if not (inv_ok and topk_ok):
        worst = key
print(json.dumps({"results": results, "any_failure": worst,
                  "metric": "kept>=dropped invariant + all entries >kth kept",
                  "reason": "contract acceptance condition is the min/max dominance test"}))
