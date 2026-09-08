import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_05/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def exact_ref(s, pivot):
    a = s[s > pivot]
    if a.numel() == 0: return 0
    return int((a == a.min()).sum().item())

torch.manual_seed(0)
out = {}
for name, gen, pivot_fn in [
    ("logits_N2(0,2)_N1024", lambda: torch.randn(1024, device="cuda")*2.0, lambda s: float(s.median())),
    ("probs_softmax_N1024", lambda: torch.softmax(torch.randn(1024, device="cuda")*2.0, dim=0), lambda s: float(s.median())),
    ("probs_softmax_tailpivot", lambda: torch.softmax(torch.randn(1024, device="cuda")*2.0, dim=0), lambda s: 0.0),
    ("logits_N256_scaleNone", lambda: torch.randn(256, device="cuda"), lambda s: float(s.median())),
]:
    over = 0; trials = 200; maxover = 0; sum_got = 0; sum_ref = 0
    for _ in range(trials):
        s = gen().contiguous()
        p = pivot_fn(s)
        got = m.count_tied_at_boundary(s, p)
        ref = exact_ref(s, p)
        sum_got += got; sum_ref += ref
        if got > ref:
            over += 1; maxover = max(maxover, got - ref)
    out[name] = {"trials": trials, "overcount_trials": over,
                 "overcount_frac": over/trials, "max_overcount": maxover,
                 "mean_got": sum_got/trials, "mean_exact_ref": sum_ref/trials,
                 "pivot_rule": "median" if "tailpivot" not in name else "0.0"}
# how far apart are the two smallest above-pivot values typically (logit case)?
gaps = []
for _ in range(200):
    s = (torch.randn(1024, device="cuda")*2.0)
    a = torch.sort(s[s > float(s.median())]).values
    if a.numel() >= 2: gaps.append(float(a[1]-a[0]))
gaps_t = torch.tensor(gaps)
out["logit_boundary_gap_stats"] = {"median_gap": float(gaps_t.median()),
                                  "frac_gap_lt_1e-3": float((gaps_t < 1e-3).float().mean()),
                                  "min_gap": float(gaps_t.min())}
print(json.dumps(out))
