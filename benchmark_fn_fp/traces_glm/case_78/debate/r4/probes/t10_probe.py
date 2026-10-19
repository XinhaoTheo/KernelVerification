import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel

dev = "cuda"
results = {}
try:
    rows = []
    for j in range(64):
        r = torch.zeros(64, dtype=torch.int32); r[j] = 1; rows.append(r)
        r2 = torch.zeros(64, dtype=torch.int32); r2[j] = 3328; rows.append(r2)
    rows.append(torch.full((64,), 3328, dtype=torch.int32))
    g = torch.Generator().manual_seed(11)
    for _ in range(20):
        rows.append(torch.randint(0, 3329, (64,), generator=g, dtype=torch.int32))
    x = torch.stack(rows).to(dev)
    out = kernel.run(x)
    ref = kernel.reference(x)
    eq = torch.equal(out, ref)
    mismatch = {"num_rows": int((~torch.eq(out, ref).all(dim=1)).sum()), "total": int((out != ref).sum())}
    if not eq:
        bad = (~(out == ref)).any(dim=1).nonzero()[0].item()
        mismatch["first_bad_row"] = bad
        mismatch["out_row"] = out[bad][:8].tolist()
        mismatch["ref_row"] = ref[bad][:8].tolist()
    results = {"n": 64, "rows_tested": x.shape[0], "exact_equal": bool(eq), "mismatch": mismatch}
except Exception as e:
    results = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps(results))