import json, torch, sys
sys.path.insert(0, "/root/cases/case_26")
from kernel import topk_mask

torch.manual_seed(0)
results = []
worst = []
for N, K, dist in [(256, 10, "randn"), (1024, 50, "randn"),
                   (64, 32, "clustered"), (256, 10, "tied"),
                   (256, 10, "cutoff_cluster"),
                   (256, 10, "eps_close"),
                   (256, 128, "k_half")]:
    if dist == "randn":
        s = torch.randn(64, N, device="cuda")
    elif dist == "clustered":
        s = torch.randn(64, N, device="cuda") * 0.01
    elif dist == "tied":
        s = torch.randint(0, 5, (64, N), device="cuda").float()
    elif dist == "cutoff_cluster":
        s = torch.randn(64, N, device="cuda")
        s[:, 5:40] = s[:, 5:6]  # many values equal near cutoff
    elif dist == "eps_close":
        # 256 distinct values, spacing ~1e-7 around cutoff: stresses bisection precision
        base = torch.sort(torch.randn(N, device="cuda"))[0]
        s = base.unsqueeze(0).repeat(64, 1).contiguous()
    else:  # k_half
        s = torch.randn(64, N, device="cuda")

    out = topk_mask(s, K)
    torch.cuda.synchronize()
    inv_ok = True; topk_ok = True; overkept = 0
    kept_counts = []
    for r in range(s.shape[0]):
        row = s[r]; o = out[r]
        kept = o != 0; dropped = ~kept
        kept_counts.append(int(kept.sum()))
        if kept.sum() > 0 and dropped.sum() > 0:
            if o[kept].min() < row[dropped].max():
                inv_ok = False
        kth = torch.topk(row, min(K, N)).values.min()
        if (o[row > kth] == 0).any():
            topk_ok = False
        # count kept entries strictly below kth (untied extra keeps)
        extra = int(((o != 0) & (row < kth - 0)).sum())
        overkept += extra
    key = f"{N}/{K}/{dist}"
    ok = inv_ok and topk_ok
    results.append({"case": key, "invariant_kept_ge_dropped": inv_ok,
                    "all_above_kth_kept": topk_ok,
                    "kept_not_tied_below_kth": overkept,
                    "kept_count_range": [min(kept_counts), max(kept_counts)]})
    if not ok:
        worst.append(key)
print(json.dumps({"results": results, "failures": worst,
                  "metric": "kept>=dropped dominance + all entries >kth kept + untied extra-keep count",
                  "reason": "contract acceptance condition is min(kept)>=max(dropped); untied extras below kth would violate top-k beyond the admitted tie reading"}))