
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_26/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

res = {}
def analyze(name, row, k, dt):
    x = torch.tensor([row], dtype=dt, device="cuda")
    r = [float(v) for v in x[0].tolist()]
    out = [float(v) for v in m.topk_mask(x, k)[0].tolist()]
    kept = [i for i in range(len(r)) if out[i] == r[i] and r[i] != 0.0]
    v = torch.topk(torch.tensor(r, dtype=torch.float64), k).values[-1].item()
    below = [r[i] for i in kept if r[i] < v]
    res[name] = {
        "dtype": str(dt), "n": len(r), "k": k, "kth_largest": v,
        "kept_count": len(kept),
        "n_kept_strictly_below_kth": len(below),
        "max_kept_below_kth": max(below) if below else None,
        "min_kept_below_kth": min(below) if below else None,
        "ties_at_kth_value": sum(1 for val in r if val == v),
        "row_min": min(r), "row_max": max(r),
    }

# bf16 near-cutoff neighbours a few ulps apart, wide row range
base = 1.0
bf = torch.tensor([base*(1+ i*0.0078125) for i in range(12)], dtype=torch.bfloat16)
row_bf = [100.0] + [float(v) for v in bf.tolist()]
analyze("bf16_near_cutoff", row_bf, 4, torch.bfloat16)

# fp16 same idea
fh = torch.tensor([1.0 + i*0.0009765625 for i in range(12)], dtype=torch.float16)
row_fh = [100.0] + [float(v) for v in fh.tolist()]
analyze("fp16_near_cutoff", row_fh, 4, torch.float16)

# fp16 overflow: lo+hi > 65504
row_ovf = [30000.0, 32000.0, 35000.0, 40000.0, 45000.0, 50000.0, 31000.0, 33000.0]
analyze("fp16_overflow_row", row_ovf, 2, torch.float16)
print(json.dumps(res))
