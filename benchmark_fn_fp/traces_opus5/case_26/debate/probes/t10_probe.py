
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
def analyze(name, row, k):
    x = torch.tensor([row], dtype=torch.float32, device="cuda")
    r = x[0].tolist()
    out = m.topk_mask(x, k)[0].tolist()
    kept = [i for i in range(len(r)) if out[i] == r[i] and r[i] != 0.0]
    v = torch.topk(torch.tensor(r, dtype=torch.float64), k).values[-1].item()
    below = [r[i] for i in kept if r[i] < v]
    ties_at_v = sum(1 for val in r if val == v)
    res[name] = {
        "n": len(r), "k": k, "kth_largest": v,
        "kept_count": len(kept),
        "n_kept_strictly_below_kth": len(below),
        "max_kept_below_kth": max(below) if below else None,
        "min_kept_below_kth": min(below) if below else None,
        "ties_at_kth_value": ties_at_v,
        "row_min": min(r), "row_max": max(r),
        "bisect_resolution_est": (max(r)-min(r))*2**-40,
    }

# large outlier + cluster of small distinct scores spaced 1e-9
row1 = [1e6] + [1e-3 + i*1e-9 for i in range(20)]
analyze("outlier_1e6_cluster", row1, 5)

# softmax-like weights: max ~1, cutoff ~1e-8, neighbours spaced ~1e-12
row2 = [1.0, 0.3, 0.05] + [1e-8 + i*1e-12 for i in range(16)]
analyze("softmax_like_tail", row2, 6)

# plain randn control, well-scaled
g = torch.Generator().manual_seed(0)
row3 = torch.randn(64, generator=g, dtype=torch.float32).tolist()
analyze("randn_control", row3, 8)
print(json.dumps(res))
