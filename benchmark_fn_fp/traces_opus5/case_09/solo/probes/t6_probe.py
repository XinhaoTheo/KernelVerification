
import importlib.util, torch, json
spec = importlib.util.spec_from_file_location("kmod", "/root/cases/case_10/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
out = {}

def check(name, scores, k):
    s = scores.cuda().contiguous()
    idx = m.sorted_topk_indices(s.clone(), k).cpu()
    ref = torch.argsort(s.cpu(), dim=1, descending=True, stable=True)[:, :k]
    vals_k = torch.gather(s.cpu(), 1, idx)
    vals_r = torch.gather(s.cpu(), 1, ref)
    val_match = torch.equal(vals_k, vals_r)
    idx_match = torch.equal(idx, ref)
    # set-level: which lower-index tied candidates were dropped
    dropped = []
    for b in range(s.shape[0]):
        sel = set(idx[b].tolist()); rsel = set(ref[b].tolist())
        miss = sorted(rsel - sel); extra = sorted(sel - rsel)
        if miss or extra:
            dropped.append({"row": b, "ref_only": miss, "kernel_only": extra,
                            "ref_only_vals": [s[b, i].item() for i in miss],
                            "kernel_only_vals": [s[b, i].item() for i in extra]})
    out[name] = {"k": k, "idx_exact_match": idx_match, "selected_values_match": val_match,
                 "set_diffs": dropped[:4],
                 "kernel_idx_row0": idx[0].tolist(), "ref_idx_row0": ref[0].tolist(),
                 "scores_row0": s[0].cpu().tolist()}

# 1) all equal row -> contract demands 0..k-1
check("all_equal_N16", torch.zeros(2,16), 8)
# 2) quantized scores, many ties
check("quantized_3levels_N16", torch.randint(0,3,(4,16)).float(), 8)
# 3) distinct scores sanity
check("distinct_N16", torch.randn(4,16), 8)
# 4) larger, ties at the cutoff
s = torch.arange(32).float().flip(0).repeat(2,1)/4.0  # duplicates in groups of 4
check("groups_of_4_N32", s, 8)
print(json.dumps(out, indent=1))
