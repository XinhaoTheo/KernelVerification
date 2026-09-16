
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
res = {}

def run(name, X, k):
    X = X.cuda().contiguous()
    O = m.topk_mask(X, k)
    Xc = X.double().cpu(); Oc = O.double().cpu()
    nrows = X.shape[0]
    tot_viol = 0; rows_viol = 0; kept_counts = []; worst_gap = 0.0; worst_ratio = 0.0
    for i in range(nrows):
        r = Xc[i]; o = Oc[i]
        kept = (o == r) & (r != 0)
        v = torch.topk(r, k).values[-1].item()
        kv = r[kept]
        below = kv[kv < v]
        kept_counts.append(int(kept.sum()))
        if below.numel() > 0:
            rows_viol += 1
            tot_viol += int(below.numel())
            gap = v - below.min().item()
            worst_gap = max(worst_gap, gap)
            worst_ratio = max(worst_ratio, gap / abs(v) if v != 0 else float("inf"))
    res[name] = {
        "rows": nrows, "n_cols": X.shape[1], "k": k,
        "rows_with_violation": rows_viol,
        "total_kept_below_kth": tot_viol,
        "max_kept_count": max(kept_counts), "min_kept_count": min(kept_counts),
        "worst_abs_gap_below_kth": worst_gap,
        "worst_rel_gap_below_kth": worst_ratio,
        "row_max_mean": float(X.float().max(dim=1).values.mean()),
        "row_min_mean": float(X.float().min(dim=1).values.mean()),
    }

# 1. raw logits, typical LLM scale
logits = torch.randn(64, 512, dtype=torch.float32) * 3.0
run("raw_logits_randn3", logits, 50)

# 2. softmax probabilities, temperature 1
probs = torch.softmax(logits, dim=1)
run("softmax_T1_probs", probs, 50)

# 3. peaked softmax (low temperature) -> max ~1, tail ~1e-10
peaked = torch.softmax(logits / 0.1, dim=1)
run("softmax_T0.1_probs", peaked, 50)

# 4. very peaked: one dominant token
vp = torch.softmax(logits / 0.03, dim=1)
run("softmax_T0.03_probs", vp, 50)

# 5. logits with a single large outlier score (plausible after reward/bias add)
out = logits.clone(); out[:, 0] = 1e5
run("logits_with_1e5_outlier", out, 50)
print(json.dumps(res))
