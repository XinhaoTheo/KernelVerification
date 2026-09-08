
import torch, importlib.util, json, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = {}

def check(scores, k, tag):
    scores = scores.cuda().contiguous()
    out = m.topk_mask(scores, k)
    finite_ref_ok = True
    kth = scores.topk(k, dim=1).values[:, -1:]
    kept = out != 0
    cnt = kept.sum(dim=1)
    # entries kept that are strictly below the k-th largest (contract violation: must be 0)
    bad = (kept & (scores < kth))
    nbad = int(bad.sum())
    # how far below the cutoff is the worst wrongly-kept entry
    worst_gap = float((kth - torch.where(bad, scores, kth)).max()) if nbad else 0.0
    big = 3.4e38
    minkept = torch.where(kept, scores, torch.full_like(scores, big)).min(dim=1).values
    maxdrop = torch.where(~kept, scores, torch.full_like(scores, -big)).max(dim=1).values
    res[tag] = dict(k=k, N=scores.shape[1],
                    kept_count_min=int(cnt.min()), kept_count_max=int(cnt.max()),
                    kept_strictly_below_kth=nbad,
                    worst_value_gap_below_cutoff=worst_gap,
                    kth_value=float(kth[0]),
                    invariant_min_kept_ge_max_dropped=bool((minkept>=maxdrop).all()))

N = 256
# (a) -inf masked row (standard vocabulary masking in top-k sampling)
s = torch.randn(4, N)
s[:, 128:] = -float('inf')
check(s, 5, "neg_inf_masked_half")

# (b) finite outlier of varying magnitude on top of randn
for mag in [1e3, 1e6, 1e9, 1e12, 1e15, 1e20]:
    s = torch.randn(4, N)
    s[:, 0] = mag
    check(s, 5, f"outlier_{mag:g}")

# (c) plain randn scaled to huge magnitude (self-consistent scale)
for mag in [1e6, 1e20]:
    check(torch.randn(4, N)*mag, 5, f"scaled_randn_{mag:g}")

# (d) very peaked softmax probabilities (tiny tail values, range ~1)
logits = torch.randn(4, N)*40
check(torch.softmax(logits, dim=1), 5, "peaked_softmax_T40")
check(torch.softmax(logits, dim=1), 50, "peaked_softmax_T40_k50")

# (e) all-equal row
check(torch.full((4, N), 2.0), 5, "all_equal")
print(json.dumps(res, indent=1))
