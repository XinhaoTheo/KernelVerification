import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel
dev = "cuda"
results = {}
try:
    rows = []
    for j in range(128):
        r = torch.zeros(128, dtype=torch.int32); r[j] = 1; rows.append(r)
    x = torch.stack(rows).to(dev)  # exactly 128 rows
    out = kernel.run(x)
    ref = kernel.reference(x)
    eq = torch.equal(out, ref)
    bad = (~(out == ref)).any(dim=1)
    m = {"num_rows_mismatch": int(bad.sum()), "total_cells": int((out != ref).sum())}
    if not eq:
        b = bad.nonzero()[0].item()
        m["first_bad_row"] = b
        m["out_row"] = out[b][:8].tolist(); m["ref_row"] = ref[b][:8].tolist()
    results = {"n": 128, "rows_tested": 128, "exact_equal": bool(eq), "mismatch": m}
except Exception as e:
    results = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps(results))