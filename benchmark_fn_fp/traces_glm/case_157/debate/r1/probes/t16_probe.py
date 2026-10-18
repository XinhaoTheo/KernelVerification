
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_157/kernel.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
dev = "cuda"
C = 32  # r = 0..31
r = torch.arange(C, dtype=torch.int32, device=dev)
z = torch.zeros(C, dtype=torch.int32, device=dev)
rows = 130
m_vals = [2**31 - 1 - 2*((c*13) % 7) for c in range(C)]  # odd multipliers
m = torch.tensor(m_vals, dtype=torch.int32).to(dev)
rows_list = []
for c in range(C):
    mv, rc = m_vals[c], c
    mod = 1 << (31 + rc)
    vals = []
    if mv % 2 == 1:
        minv = pow(mv, -1, mod)
        a_half = (1 << (30 + rc)) * minv % mod   # exact halfway: a*m == 2^(30+r) mod 2^(31+r)
        for v in (a_half, a_half - 1, a_half + 1):
            if -(1 << 31) <= v < (1 << 31):
                vals.append(v)
                nv = v - (1 << 32)
                if -(1 << 31) <= nv < (1 << 31):
                    vals.append(nv)
    # nonsaturating targets: R in [-100,100]
    denom = mod
    for q in range(-100, 101, 20):
        a_base = int(round(q * denom / mv))
        for d in (-1, 0, 1):
            if -(1 << 31) <= a_base + d < (1 << 31):
                vals.append(a_base + d)
    while len(vals) < rows: vals.append(0)
    rows_list.append(vals[:rows])
a = torch.tensor(rows_list, dtype=torch.int64).t().contiguous().to(torch.int32).to(dev)
# extreme rows: a = -2^31 and 2^31-1 across all r
a[0, :] = -(1 << 31)
a[1, :] = (1 << 31) - 1
out = K.run(a, m, r, z)
ref = K.reference(a, m, r, z)
mism = (out != ref)
n = int(mism.sum())
per_r = {}
if n:
    for i, c in mism.nonzero().tolist():
        per_r.setdefault(c, []).append({"a": int(a[i, c]), "m": int(m[c]), "r": c,
                                        "expected": int(ref[i, c]), "got": int(out[i, c])})
nonsat = int((((ref > -128) & (ref < 127)) & mism).sum())
print(json.dumps({"claim": "c3", "mismatch_count": n, "total": a.numel(),
                  "nonsaturated_mismatches": nonsat,
                  "extreme_row_a=-2^31_mismatches": int(mism[0].sum()),
                  "extreme_row_a=2^31-1_mismatches": int(mism[1].sum()),
                  "per_channel_r_examples": {str(k): v[:4] for k, v in per_r.items()}}))
