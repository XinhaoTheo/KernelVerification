import torch, sys, json
sys.path.insert(0, "/root/cases/case_141")
import kernel
vals = torch.arange((1 << 32) - 100000, (1 << 32), dtype=torch.int64)
x = vals.to(torch.uint32).cuda()
out = kernel.run(x).cpu().to(torch.int64)
v = x.cpu().to(torch.int64)
col = out[:, 1]
row = out[:, 0]
recon = 7 * row + col  # invariant in uint32 arithmetic sense
recon32 = ((7 * row + col) % (1 << 32))
invariant_ok = recon32 == v
col_in_range = (col >= 0) & (col <= 6)
bad = (~col_in_range).nonzero().flatten()
import json
res = {
    "n_tested": int(v.numel()),
    "bad_col_count": int(bad.numel()),
    "example_vals": [int(v[i]) for i in bad[:10].tolist()],
    "example_cols": [int(col[i]) for i in bad[:10].tolist()],
    "example_rows": [int(row[i]) for i in bad[:10].tolist()],
    "invariant_violations": int((~invariant_ok).sum()),
    "max_col_observed": int(col.max()) if col.numel() else None,
}
print(json.dumps(res))