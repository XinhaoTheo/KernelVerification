
import json, importlib.util, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_32/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
res = {}
dev = "cuda"

# Case A: all-equal row, W=8 (power of two)
a = torch.full((1,8), 5, dtype=torch.int64, device=dev)
sk, p = m.triton_argsort(a)
res["allequal_W8_perm"] = p[0].tolist()
res["allequal_W8_ref_stable_perm"] = torch.sort(a, dim=1, stable=True).indices[0].tolist()
res["allequal_W8_keys_sorted_ok"] = bool((sk[0].diff() >= 0).all().item())

# Case B: small-range keys, power-of-two W, many ties
tot_rows = 0; bad_rows = 0; keys_ok_rows = 0
details = []
for W in [8, 16, 32, 64]:
    N = 16
    k = torch.randint(0, 4, (N, W), dtype=torch.int64, device=dev)
    sk, p = m.triton_argsort(k)
    ref = torch.sort(k, dim=1, stable=True)
    row_bad = (p != ref.indices).any(dim=1)
    keys_match = (sk == ref.values).all(dim=1)
    tot_rows += N; bad_rows += int(row_bad.sum().item()); keys_ok_rows += int(keys_match.sum().item())
    details.append({"W": W, "perm_mismatch_rows": int(row_bad.sum().item()),
                    "keys_match_rows": int(keys_match.sum().item()), "rows": N})
res["pow2_tied_detail"] = details
res["pow2_total_rows"] = tot_rows
res["pow2_perm_mismatch_rows"] = bad_rows
res["pow2_keys_match_rows"] = keys_ok_rows

# Case C: check whether ties come out in strictly DESCENDING index order (anti-stable)
k = torch.randint(0, 3, (8, 16), dtype=torch.int64, device=dev)
sk, p = m.triton_argsort(k)
anti = torch.sort(k * 1000 - torch.arange(16, device=dev), dim=1, stable=True).indices  # ties -> descending idx
res["antistable_perm_match_rows"] = int((p == anti).all(dim=1).sum().item())
res["antistable_rows_total"] = 8
res["example_row_keys"] = k[0].tolist()
res["example_row_perm"] = p[0].tolist()
res["example_row_stable_ref"] = torch.sort(k, dim=1, stable=True).indices[0].tolist()
print(json.dumps(res))
