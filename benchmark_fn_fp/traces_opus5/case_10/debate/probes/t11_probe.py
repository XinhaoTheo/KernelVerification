
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_10/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref_stable(scores, k):
    v, i = torch.sort(scores.cpu().float(), dim=-1, descending=True, stable=True)
    return i[:, :k]

res = {"cases": []}

def run(name, scores, k):
    sc = scores.cuda()
    out = m.sorted_topk_indices(sc, k).cpu()
    ref = ref_stable(sc, k)
    sel = torch.gather(scores, 1, out)
    refv = torch.gather(scores, 1, ref)
    finite_mismatch = 0
    for r in range(out.shape[0]):
        a = sel[r]; b = refv[r]
        fm = ((a != b) & torch.isfinite(b)).sum().item()
        finite_mismatch += fm
    res["cases"].append({
        "name": name, "N": scores.shape[1], "k": k,
        "kernel_topk_row0": out[0].tolist(),
        "ref_topk_row0": ref[0].tolist(),
        "kernel_selected_values_row0": [None if not torch.isfinite(v) else float(v) for v in sel[0]],
        "ref_selected_values_row0": [None if not torch.isfinite(v) else float(v) for v in refv[0]],
        "index_set_matches_ref": bool(all(set(out[r].tolist()) == set(ref[r].tolist()) for r in range(out.shape[0]))),
        "n_selected_value_mismatches_where_ref_finite": finite_mismatch,
        "kernel_selected_any_neg_inf": bool(torch.isneginf(sel).any()),
        "ref_selected_any_neg_inf": bool(torch.isneginf(refv).any()),
    })

# baseline sanity: all-finite distinct row must sort exactly
torch.manual_seed(1)
run("finite_distinct_baseline_N16_k4", torch.randn(4, 16), 4)

# one -inf among finite distinct values, k excludes the -inf
row = torch.tensor([[3.0, -float('inf'), 1.0, 7.0, 5.0, -2.0, 0.5, 9.0,
                     4.0, 2.0, 6.0, -1.0, 8.0, 0.0, -3.0, 1.5]])
run("single_neg_inf_N16_k4", row, 4)

# half the row masked with -inf (typical NSA masking)
row2 = torch.linspace(0.0, 7.0, 16).unsqueeze(0).clone()
row2[0, 8:] = -float('inf')
run("half_masked_neg_inf_N16_k4", row2, 4)

# +inf present
row3 = torch.linspace(0.0, 7.0, 16).unsqueeze(0).clone()
row3[0, 5] = float('inf')
run("pos_inf_N16_k4", row3, 4)

res["metric"] = "selected values / index set of FINITE entries vs CPU stable descending reference"
print(json.dumps(res))
