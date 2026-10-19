
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
    # is kernel output ascending within each equal-value run?
    asc_ok = True
    for r in range(out.shape[0]):
        vals = sel[r].tolist(); idxs = out[r].tolist()
        run_start = 0
        for j in range(1, k + 1):
            if j == k or vals[j] != vals[run_start]:
                seg = idxs[run_start:j]
                if seg != sorted(seg):
                    asc_ok = False
                run_start = j
    res["cases"].append({
        "name": name, "k": k, "N": scores.shape[1],
        "kernel_topk_row0": out[0].tolist(),
        "ref_topk_row0": ref[0].tolist(),
        "index_sequence_matches_stable_ref": bool(torch.equal(out, ref)),
        "ascending_within_equal_runs": asc_ok,
        "selected_values_max_abs_diff": float((sel - refv).abs().max()),
        "kernel_topk_set_equals_ref_set": bool(all(set(out[r].tolist()) == set(ref[r].tolist()) for r in range(out.shape[0]))),
    })

run("all_equal_N16_k8", torch.ones(1, 16), 8)
run("all_equal_N32_k8", torch.full((1, 32), 3.5), 8)
# 6 copies of the top value fully inside top-k
row = torch.linspace(-2.0, -0.1, 16).unsqueeze(0).clone()
for j in [1, 3, 4, 8, 12, 13]:
    row[0, j] = 7.0
run("six_ties_inside_topk_N16_k8", row, 8)
# quantized/clustered row: few distinct values
torch.manual_seed(0)
q = (torch.randint(0, 3, (4, 32)).float())
run("quantized_3level_N32_k8", q, 8)

res["metric"] = "exact index sequence within each equal-score run vs CPU stable descending sort (lower-index-first)"
print(json.dumps(res))
