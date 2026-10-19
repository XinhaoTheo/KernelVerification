
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_157/kernel.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
dev = "cuda"
C = 16
m_vals = [2**31-1 - 2*((c*13) % 7) for c in range(C)]  # odd multipliers near 2^31
m = torch.tensor(m_vals, dtype=torch.int32).to(dev)
r = torch.zeros(C, dtype=torch.int32, device=dev)
z = torch.zeros(C, dtype=torch.int32, device=dev)
# engineer residuals P mod 2^31 = 2^30 + d for d in (-600,600)\{0}: the window where
# fp64 product error (up to ~2^9) can cross the k+0.5 rounding boundary
rows = 200
rows_list = []
for mv in m_vals:
    minv = pow(mv, -1, 1 << 31)
    vals = []
    for d in list(range(-600, 601, 3)):
        if d == 0: continue
        a = ((1 << 30) + d) * minv % (1 << 31)
        vals.append(a)
        if a >= (1 << 30):
            vals.append(a - (1 << 31))  # negative twin, |P| ~ 2^61
    vals.append((1 << 30) * minv % (1 << 31))
    while len(vals) < rows: vals.append(0)
    rows_list.append(vals[:rows])
a = torch.tensor(rows_list, dtype=torch.int64).t().contiguous().to(torch.int32).to(dev)
out = K.run(a, m, r, z)
ref = K.reference(a, m, r, z)
mism = (out != ref)
n = int(mism.sum())
examples = []
if n:
    for i, c in mism.nonzero().tolist()[:12]:
        P = int(a[i, c]) * int(m[c])
        examples.append({"a": int(a[i, c]), "m": int(m[c]),
                         "P_mod_2^31_minus_2^30": (P % (1 << 31)) - (1 << 30),
                         "expected": int(ref[i, c]), "got": int(out[i, c])})
    d = (out.to(torch.int64) - ref.to(torch.int64))[mism]
    print("max_abs_diff_mismatched:", int(d.abs().max()))
print(json.dumps({"claim": "c1", "mismatch_count": n, "total": a.numel(),
                  "engineered_window": "residual within +-600 of 2^30", "examples": examples}))
