
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k9", "/root/cases/case_09/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(7)
out = {}

# Part A: exhaustive duplicate-max pairs (i<j), rest strictly lower
partA = {}
tot_pairs = tot_high = tot_low = tot_other = 0
for N in [2, 4, 8, 16, 32]:
    subs = [(i, j) for i in range(N) for j in range(i+1, N)]
    rows = []
    for (i, j) in subs:
        r = torch.full((N,), -1.0); r[i] = 1.0; r[j] = 1.0
        rows.append(r)
    scores = torch.stack(rows).cuda()
    got = m.sorted_topk_indices(scores, 1).squeeze(1).cpu()
    lo = torch.tensor([i for (i, _) in subs]); hi = torch.tensor([j for (_, j) in subs])
    nlow = int((got == lo).sum()); nhigh = int((got == hi).sum())
    nother = int(((got != lo) & (got != hi)).sum())
    # value check: returned value still equals row max
    val_max = int((scores.cpu().gather(1, got[:, None]).squeeze(1) == scores.cpu().max(dim=1).values).sum())
    partA[str(N)] = {"pairs": len(subs), "returned_lower_index": nlow,
                     "returned_higher_index": nhigh, "returned_neither": nother,
                     "rows_where_value_still_max": val_max,
                     "example_failures": [[int(lo[t]), int(hi[t]), int(got[t])]
                                          for t in range(len(subs)) if int(got[t]) != int(lo[t])][:5]}
    tot_pairs += len(subs); tot_high += nhigh; tot_low += nlow; tot_other += nother
out["pairwise"] = partA
out["pairwise_totals"] = {"pairs": tot_pairs, "returned_higher_index": tot_high,
                          "returned_lower_index": tot_low, "returned_neither": tot_other,
                          "violation_rate": round(tot_high / max(tot_pairs, 1), 4)}

# Part B: quantized rows (natural tie source), in-domain N
partB = {}
sumB = {"rows": 0, "mismatch": 0, "higher": 0, "lower": 0, "value_still_max": 0}
for N in [2, 8, 64, 256, 1024]:
    for L in [2, 4, 8]:
        B = 256
        scores = torch.randint(0, L, (B, N), device="cuda").float()
        got = m.sorted_topk_indices(scores, 1).squeeze(1)
        ref = scores.argmax(dim=1)  # torch.argmax -> first (lowest) occurrence of row max
        mism = got != ref
        vmax = int((scores.gather(1, got[:, None]).squeeze(1) == scores.max(dim=1).values).sum())
        tied = int((scores == scores.max(dim=1, keepdim=True).values).sum(dim=1).gt(1).sum())
        rec = {"rows": B, "tied_max_rows": tied, "mismatch_rows": int(mism.sum()),
               "higher_than_ref": int((got[mism] > ref[mism]).sum()),
               "lower_than_ref": int((got[mism] < ref[mism]).sum()),
               "rows_where_value_still_max": vmax,
               "mismatch_rate": round(float(mism.float().mean()), 4)}
        partB[f"N{N}_L{L}"] = rec
        sumB["rows"] += B; sumB["mismatch"] += rec["mismatch_rows"]
        sumB["higher"] += rec["higher_than_ref"]; sumB["lower"] += rec["lower_than_ref"]
        sumB["value_still_max"] += vmax
out["quantized"] = partB
out["quantized_totals"] = sumB

# Part C: downstream mean-pooled representation
N, B, D = 64, 256, 32
scores = torch.randint(0, 4, (B, N), device="cuda").float()
experts = torch.randn(N, D, device="cuda")
got = m.sorted_topk_indices(scores, 1).squeeze(1); ref = scores.argmax(dim=1)
pg = experts[got].mean(0); pr = experts[ref].mean(0)
out["downstream"] = {"N": N, "B": B, "D": D, "misrouted_tokens": int((got != ref).sum()),
                     "pooled_max_abs_diff": float((pg - pr).abs().max()),
                     "pooled_rel_l2": float((pg - pr).norm() / pr.norm())}

# Part D: sanity that a strictly-distinct control still matches (no confound)
ctrl = torch.randn(256, 64, device="cuda")
ctrl_mism = int((m.sorted_topk_indices(ctrl, 1).squeeze(1) != ctrl.argmax(dim=1)).sum())
out["distinct_control_mismatch_rows"] = ctrl_mism

out["metric"] = "returned k=1 index vs lowest-index-tie-break argmax; direction of mismatch; value-still-max check"
print(json.dumps(out))
