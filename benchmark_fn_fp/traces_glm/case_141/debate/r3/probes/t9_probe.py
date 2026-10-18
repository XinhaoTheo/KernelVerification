import json, sys
sys.path.insert(0, "/root/cases/case_141")
import torch, kernel

vals = [4294967295]
for r in range(7):
    vals.append(4294967295 - ((4294967295 - r) % 7))
    vals.append(3435973837 + ((r - 3435973837) % 7))
vals = sorted(set(vals))
idx = torch.tensor(vals, dtype=torch.uint32, device="cuda")
out = kernel.run(idx)
out_i = out.to(torch.int64)
v64 = idx.to(torch.int64)
rows, cols = out_i[:, 0], out_i[:, 1]

col_in_range = (cols >= 0) & (cols <= 6)
recompose = (7 * rows + cols) == v64
bad = ~(col_in_range & recompose)
result = {
    "n_tested": len(vals),
    "violations": int(bad.sum()),
    "bad_values": v64[bad].tolist(),
    "bad_cols": cols[bad].tolist(),
    "bad_rows": rows[bad].tolist(),
    "uint32max_col": int(cols[vals.index(4294967295)]),
    "uint32max_row": int(rows[vals.index(4294967295)]),
}
print(json.dumps(result))