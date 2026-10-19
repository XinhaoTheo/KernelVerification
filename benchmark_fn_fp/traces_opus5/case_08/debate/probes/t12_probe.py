
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k9", "/root/cases/case_09/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(1)
out = {}
# realistic tie-producing gate scores: small integer / quantized value sets
for N in [8, 64, 256, 1024]:
    for nlev in [2, 4, 8]:
        B = 256
        scores = torch.randint(0, nlev, (B, N), device="cuda").float()
        top1 = m.sorted_topk_indices(scores, 1).squeeze(1)
        ref = scores.argmax(dim=1)  # torch.argmax = first (lowest) occurrence of max
        mism = (top1 != ref)
        # confirm mismatches are ties (same value), i.e. contract violation not value error
        val_ok = int((scores.gather(1, top1[:,None]).squeeze(1) == scores.gather(1, ref[:,None]).squeeze(1)).sum())
        tie_rows = int((scores == scores.max(dim=1, keepdim=True).values).sum(dim=1).gt(1).sum())
        out[f"N{N}_lev{nlev}"] = {"rows": B, "tied_max_rows": tie_rows,
            "index_mismatch_rows": int(mism.sum()),
            "rows_where_value_still_max": val_ok,
            "mismatch_rate": round(float(mism.float().mean()), 4),
            "higher_than_ref": int((top1[mism] > ref[mism]).sum()),
            "lower_than_ref": int((top1[mism] < ref[mism]).sum())}
# downstream: mean-pooled expert output difference
N, B, D = 64, 256, 16
scores = torch.randint(0, 4, (B, N), device="cuda").float()
experts = torch.randn(N, D, device="cuda")
got = m.sorted_topk_indices(scores, 1).squeeze(1)
ref = scores.argmax(dim=1)
pooled_got = experts[got].mean(0); pooled_ref = experts[ref].mean(0)
out["downstream"] = {"N": N, "B": B, "D": D,
    "misrouted_tokens": int((got != ref).sum()),
    "pooled_max_abs_diff": float((pooled_got - pooled_ref).abs().max()),
    "pooled_rel_l2": float((pooled_got-pooled_ref).norm() / pooled_ref.norm())}
print(json.dumps({"quantized_tie_scan": out,
  "metric": "k=1 index vs torch.argmax (lowest-index tie-break) on quantized score rows; plus mean-pooled expert-output diff"}))
