
import json, importlib.util, itertools, torch
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_10/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def sweep(N, k):
    rows, meta = [], []
    for lo, hi in itertools.combinations(range(N), 2):
        row = torch.linspace(-1.0, -0.1, N).clone()
        dom = [j for j in range(N) if j not in (lo, hi)][:k-1]
        for r, j in enumerate(dom):
            row[j] = 100.0 - r
        row[lo] = 50.0; row[hi] = 50.0
        rows.append(row); meta.append((lo, hi))
    scores = torch.stack(rows).cuda()
    out = m.sorted_topk_indices(scores, k).cpu()
    viol = []
    for r, (lo, hi) in enumerate(meta):
        got = set(out[r].tolist())
        # both tied cannot be in top-k since k-1 dominants + 1 slot
        if hi in got and lo not in got:
            viol.append([lo, hi])
        elif lo not in got and hi not in got:
            viol.append([lo, hi, "neither"])
    return len(meta), viol, out

res = {}
for (N, k) in [(16, 4), (16, 8), (32, 4)]:
    n, viol, out = sweep(N, k)
    res[f"N{N}_k{k}"] = {
        "n_placements": n,
        "n_violations_higher_index_kept": len(viol),
        "violation_rate": round(len(viol) / n, 4),
        "example_violations": viol[:12],
    }
res["metric"] = "fraction of exact tied-pair placements at the k cutoff where the LOWER index is dropped"
print(json.dumps(res))
