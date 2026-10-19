import json, torch, sys
sys.path.insert(0, "/root/cases/case_141")
import kernel as K

# targeted upper-range values: around 3435973837 and n ≡ 6 mod 7, plus max uint32
vals = [3435973836, 3435973841, 3435973843, 4294967291, 4294967295, 4000000006, 3600000000, 3435973837, 3435973838]
# ensure some ≡6 mod 7
vals += [4294967290, 4294967289, 1000000006, 2000000006]
t = torch.tensor(vals, dtype=torch.int64).to(torch.uint32).cuda()
out = K.run(t).cpu().long()
x = t.cpu().long()
exp_row = x // 7
exp_col = x % 7
row = out[:, 0].long()
col = out[:, 1].long()
mismatches = (row != exp_row) | (col != exp_col)
result = {
    "values": [int(v) for v in vals],
    "kernel_row": row.tolist(),
    "kernel_col": col.tolist(),
    "expected_row": exp_row.tolist(),
    "expected_col": exp_col.tolist(),
    "mismatch_indices": mismatches.nonzero().flatten().tolist(),
    "n_mismatch": int(mismatches.sum()),
    "col_in_range_0_6": bool(((col >= 0) & (col <= 6)).all()),
    "reconstruction_ok": bool((7*row + col == x).all()),
}
print(json.dumps(result))