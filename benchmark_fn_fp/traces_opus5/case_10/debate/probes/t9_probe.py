
import json, sys, importlib.util, torch
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_10/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref_stable(scores, k):
    v, i = torch.sort(scores.cpu().float(), dim=-1, descending=True, stable=True)
    return i[:, :k]

res = {}
N, k = 16, 4
# Row construction: 3 strictly-largest values, then an exact tie for the 4th/5th slot
# placed at (low_idx, high_idx), then strictly smaller filler.
cases = []
for (lo, hi) in [(5, 11), (3, 4), (0, 15), (6, 7), (2, 9)]:
    row = torch.full((N,), 0.0)
    filler = torch.linspace(-1.0, -0.1, N)
    row.copy_(filler)
    # three dominant
    dom = [j for j in range(N) if j not in (lo, hi)][:3]
    for r, j in enumerate(dom):
        row[j] = 100.0 - r
    row[lo] = 50.0
    row[hi] = 50.0
    cases.append((row, lo, hi, sorted(dom)))

scores = torch.stack([c[0] for c in cases]).cuda()
out = m.sorted_topk_indices(scores, k).cpu()
ref = ref_stable(scores, k)

rows = []
n_lower_kept = 0
for r, (row, lo, hi, dom) in enumerate(cases):
    got = set(out[r].tolist())
    rows.append({
        "lo_idx": lo, "hi_idx": hi,
        "kernel_topk": out[r].tolist(),
        "ref_topk": ref[r].tolist(),
        "lower_in_kernel_topk": lo in got,
        "higher_in_kernel_topk": hi in got,
    })
    if lo in got and hi not in got:
        n_lower_kept += 1
res["cutoff_tie_cases"] = rows
res["n_cases"] = len(cases)
res["n_cases_lower_index_kept"] = n_lower_kept
res["n_cases_higher_index_kept"] = len(cases) - n_lower_kept

# selected values must still be value-correct (weak proxy check)
sel_vals = torch.gather(scores.cpu(), 1, out)
ref_vals = torch.gather(scores.cpu(), 1, ref)
res["selected_values_max_abs_diff"] = float((sel_vals - ref_vals).abs().max())

# downstream: gather distinct value vectors and average
D = 8
V = torch.arange(N * D, dtype=torch.float32).reshape(N, D)
avg_k = V[out].mean(dim=1)
avg_r = V[ref].mean(dim=1)
res["downstream_avg_max_abs_diff"] = float((avg_k - avg_r).abs().max())
res["metric"] = "membership of tied index pair in returned top-k, plus downstream averaged-vector diff"
print(json.dumps(res))
