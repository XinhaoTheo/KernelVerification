
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = {}

def check(s, k, tag):
    s = s.cuda().contiguous().float()
    out = m.topk_mask(s, k)
    kth = s.topk(k, dim=1).values[:, -1:]
    ref = torch.where(s >= kth, s, torch.zeros_like(s))
    kept = out != 0
    cnt = kept.sum(dim=1)
    bad = kept & (s < kth)                      # kept but strictly below k-th largest
    nbad = int(bad.sum())
    # ratio of worst wrongly-kept value to the cutoff (how non-tied it is)
    worst_ratio = float((torch.where(bad, s, kth) / kth).min()) if nbad else 1.0
    nonfinite_out = int((~torch.isfinite(out)).sum())
    res[tag] = dict(k=k, N=s.shape[1], kept_min=int(cnt.min()), kept_max=int(cnt.max()),
                    kept_strictly_below_kth=nbad,
                    worst_kept_over_cutoff_ratio=worst_ratio,
                    nonfinite_entries_in_out=nonfinite_out,
                    ref_all_finite=bool(torch.isfinite(ref).all()),
                    kth_row0=float(kth[0]), max_row0=float(s[0].max()))

V = 32000
# (1) ONE banned token masked with -inf in an otherwise ordinary logit row
lg = torch.randn(4, V)
lg[:, 7] = -float('inf')
check(lg, 50, "logits_one_neg_inf_mask_k50")

# (2) realistic 5% banned-token mask
lg2 = torch.randn(4, V)
mask = torch.rand(4, V) < 0.05
lg2[mask] = -float('inf')
check(lg2, 50, "logits_5pct_neg_inf_mask_k50")

# (3) peakedness scan on probabilities: how extreme must it be?
for sd in [12, 16, 20, 25, 30, 40]:
    p = torch.softmax(torch.randn(4, V) * sd, dim=1)
    check(p, 50, f"probs_logitstd{sd}_k50")

# (4) low-temperature sampling: logits std 3, temperature T
for T in [0.5, 0.2, 0.1, 0.05]:
    p = torch.softmax(torch.randn(4, V) * 3 / T, dim=1)
    check(p, 50, f"probs_std3_T{T}_k50")
print(json.dumps(res, indent=1))
