import torch, sys, json
sys.path.insert(0, "/root/cases/case_151")
import kernel

dev = "cuda"
results = {}
mismatch = None
try:
    # spike rows, constant rows, all-max rows, random canonical rows at N=128
    rows = []
    for j in range(128):
        r = torch.zeros(128, dtype=torch.int32); r[j] = 1; rows.append(r)
        r2 = torch.zeros(128, dtype=torch.int32); r2[j] = 3328; rows.append(r2)
    rows.append(torch.full((128,), 3328, dtype=torch.int32))
    g = torch.Generator().manual_seed(7)
    for _ in range(20):
        rows.append(torch.randint(0, 3329, (128,), generator=g, dtype=torch.int32))
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
    results = {"n": 128, "rows_tested": x.shape[0], "exact_equal": bool(eq), "mismatch": mismatch}
except Exception as e:
    results = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps(results))