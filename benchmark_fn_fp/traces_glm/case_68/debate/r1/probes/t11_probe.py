import torch, sys, json
sys.path.insert(0, "/root/cases/case_141")
import kernel
# mid-range values where SHIFT=34 math is provably exact (error < floor margin):
# error bound 5n/(7*2^34) < 1 - frac; safe for all n < ~8.6e9? Use n < 1e9 plus small values.
vals = torch.cat([
    torch.arange(0, 1000, dtype=torch.int64),
    torch.randint(0, 1_000_000_000, (50000,), dtype=torch.int64),
])
x = vals.to(torch.uint32).cuda()
out = kernel.run(x).cpu().to(torch.int64)
v = x.cpu().to(torch.int64)
row_ok = (out[:, 0] == torch.div(v, 7, rounding_mode="floor")).all().item()
col_ok = (out[:, 1] == v % 7).all().item()
res = {
    "n_tested": int(v.numel()),
    "max_val_tested": int(v.max()),
    "row_exact": bool(row_ok),
    "col_exact": bool(col_ok),
    "all_exact": bool(row_ok and col_ok),
}
print(json.dumps(res))