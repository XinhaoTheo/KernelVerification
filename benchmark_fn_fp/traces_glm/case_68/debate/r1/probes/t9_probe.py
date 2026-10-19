import torch, sys
sys.path.insert(0, "/root/cases/case_141")
import kernel

M, S = kernel._MULTIPLIER, kernel._SHIFT
# build sweep: values near top of uint32 range, all residues mod 7, plus some mid-range
base = (1 << 32) - 70001
vals = []
for i in range(0, 70000, 7):  # residues 0..6 automatically via +k below
    pass
for i in range(base, (1 << 32)):
    vals.append(i)
# also targeted residue-6 near top
for r in range(7):
    n = (1 << 32) - 1 - ((1 << 32) - 1 - r) % 7
vals = torch.arange((1 << 32) - 100000, (1 << 32), dtype=torch.int64)
x = vals.to(torch.uint32).cuda()
out = kernel.run(x).cpu().to(torch.int64)
v = x.cpu().to(torch.int64)
row_ok = out[:, 0] == torch.div(v, 7, rounding_mode="floor")
row_diff = (out[:, 0] - torch.div(v, 7, rounding_mode="floor")).unique()
bad = (~row_ok).nonzero().flatten()
print(torch.__version__)
import json
res = {
    "n_tested": int(v.numel()),
    "min_val": int(v.min()), "max_val": int(v.max()),
    "bad_row_count": int(bad.numel()),
    "row_diff_values": [int(d) for d in row_diff.tolist()],
    "example_bad_vals": [int(v[i]) for i in bad[:10].tolist()],
    "example_bad_rows": [int(out[i, 0]) for i in bad[:10].tolist()],
    "expected_rows": [int(v[i] // 7) for i in bad[:10].tolist()],
    "residues_of_bad": sorted(set(int(v[i] % 7) for i in bad[:200].tolist())),
}
print(json.dumps(res))