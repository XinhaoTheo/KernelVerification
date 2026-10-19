import json, torch, sys
sys.path.insert(0, "/root/cases/case_142")
import kernel

res = {}
for N in (32, 64, 128):
    # feed one-hot rows: row b = e_b, output row b should equal W_N[:, b] (basis rows' column b)
    try:
        x = torch.eye(N, dtype=torch.int16).cuda()
        out = kernel.run(x).cpu()
        ref = kernel.reference(x).cpu()
        # out[k,b] must equal ref[k,b]
        eq = bool(torch.equal(out, ref))
        # identify which rows differ and whether it's a permutation of ref
        is_perm = bool(torch.equal(out.sort(dim=0).values, ref.sort(dim=0).values))
        res[f"N{N}"] = {"onehot_exact_match": eq, "equal_as_permutation_columns": is_perm,
                        "mismatch": int((out != ref).sum().item())}
    except Exception as e:
        res[f"N{N}"] = {"error": repr(e)}
print(json.dumps(res))
