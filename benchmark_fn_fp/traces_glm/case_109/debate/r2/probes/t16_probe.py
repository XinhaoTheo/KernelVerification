import sys, json, math, torch
sys.path.insert(0, "/root/cases/case_109")
import kernel
dev = "cuda"

def is_tie(a, b):
    p = float(a) * float(b)
    r = torch.tensor(p, dtype=torch.float16).item()
    if p == r:
        return False
    inf = torch.tensor(math.inf, dtype=torch.float16)
    ninf = torch.tensor(-math.inf, dtype=torch.float16)
    rt = torch.tensor(r, dtype=torch.float16)
    if p > r:
        dn = r; up = torch.nextafter(rt, inf).item()
    else:
        dn = torch.nextafter(rt, ninf).item(); up = r
    return (dn < p < up) and (p - dn == up - p)

out = {}

# ---- Part A: M=1 engineered exact-tie columns for h-cast (fp32->fp16) and t (fp16*fp16) ----
N = 512
TIEr = 1.0 + 2 ** -10 + 2 ** -11   # x=1 => product exactly halfway, lower mantissa ODD => RN rounds UP
def rstd_of(xrow):
    m = (xrow.double() ** 2).mean().item()
    return 1.0 / math.sqrt(m + 1e-5)

base = torch.ones(N)
for i, k in enumerate([-3, -2, -1, 0, 1, 2, 3]):
    base[i] = 2.0 ** k
cands = []
v = torch.tensor(0.4, dtype=torch.float16)
hi = torch.tensor(math.inf, dtype=torch.float16)
while float(v) < 0.6:
    cands.append(float(v)); v = torch.nextafter(v, hi)
best = None
for a in cands:  # single tuning column
    x = base.clone(); x[7] = a
    if abs(rstd_of(x) - TIEr) / TIEr <= 1e-6:
        best = (a, None); break
if best is None:
    for a in cands:
        for b in cands:
            x = base.clone(); x[7] = a; x[8] = b
            if abs(rstd_of(x) - TIEr) / TIEr <= 1e-6:
                best = (a, b); break
        if best: break
assert best, "rstd tuning failed"
x = base.clone(); x[7] = best[0]
x[8] = best[1] if best[1] is not None else 1.0
r_true = rstd_of(x)
rstd = torch.full((1,), TIEr, dtype=torch.float32)
# t-multiply tie columns: x=1 -> h known; search dy with exact dy*h tie
h1 = (torch.tensor([1.0], dtype=torch.float16).float() * torch.tensor([TIEr], dtype=torch.float32)).to(torch.float16).item()
dy = torch.ones(N)
tie_dy = []
v = torch.tensor(0.25, dtype=torch.float16)
while float(v) <= 4.0:
    if is_tie(float(v), h1):
        tie_dy.append(float(v))
    v = torch.nextafter(v, hi)
for j, dv in enumerate(tie_dy[:24]):
    dy[9 + j] = dv
xh = x.half(); dyh = dy.half(); wh = torch.ones(N, dtype=torch.float16)
T = tuple(t.to(dev) for t in (xh, wh, dyh, rstd))
dx, dw = kernel.run(*T)
# expected values via torch (IEEE RN) semantics
h_t = (xh.float() * rstd[:, None]).to(torch.float16)
t_t = (dyh * h_t).to(torch.float16)  # torch fp16 multiply (RN)
dw_exp = t_t[0].double()  # M=1: dw[j] == t[0,j] exactly
diff = (dw.double() - dw_exp)
n_mm = int((diff != 0).sum())
max_abs = float(diff.abs().max()) if N else 0.0
rat = kernel.error_ratios((dx, dw), T)
out["partA"] = dict(
    rstd_tie_const=TIEr, r_true=r_true, rstd_rel_err=abs(r_true - TIEr) / TIEr,
    row_rms=float(xh.float().square().mean().sqrt()),
    h_expected_for_x1=h1, truncation_would_give=math.nextafter(h1, -math.inf) if False else None,
    t_tie_dy_count=len(tie_dy), dw_mismatch_cols=n_mm,
    dw_max_abs_diff=max_abs, dw_ratio=rat["dw"], dx_ratio=rat["dx"],
)

# ---- Part B: M=1 random exact dw == t comparison across N values ----
mm_total = 0; cols_total = 0; worst = 0.0
for Nv in (16, 17, 100, 257, 512):
    for seed in range(10):
        g = torch.Generator("cpu").manual_seed(seed * 100 + Nv)
        x = (2 * torch.rand((1, Nv), generator=g) - 1).half()
        w = (0.5 + 1.5 * torch.rand((Nv,), generator=g)).half()
        dyv = (2 * torch.rand((1, Nv), generator=g) - 1).half()
        rs = torch.rsqrt(x.float().square().mean(1) + 1e-5).float()
        T = tuple(t.to(dev) for t in (x, w, dyv, rs))
        dx, dw = kernel.run(*T)
        h_t = (x.float() * rs[:, None]).to(torch.float16)
        t_t = (dyv * h_t).to(torch.float16)
        d = (dw.double() - t_t[0].double())
        mm_total += int((d != 0).sum()); cols_total += Nv
        worst = max(worst, float(d.abs().max()))
out["partB"] = dict(random_m1_trials=50, mismatch_cols=mm_total, total_cols=cols_total, max_abs_diff=worst)

# ---- Part C: engineered exact-tie m = dy*weight columns; effect on dx ----
# tie family: (1+2^-10, 1.5): i+j odd, ij*2^-20 == 2^-11 -> exact tie, RN rounds UP
Nc = 512
x = torch.ones((1, Nc), dtype=torch.float16)
rs = torch.rsqrt(x.float().square().mean(1) + 1e-5).float()
dyv = torch.ones(Nc); wv = torch.ones(Nc)
pairs = []
for k in range(-3, 4):
    a = (2.0 ** k) * (1.0 + 2 ** -10); b = (2.0 ** k) * 1.5
    if is_tie(a, b) and abs(a) <= 4 and abs(b) <= 2:
        pairs.append((a, b)); pairs.append((b, a))
for j, (a, b) in enumerate(pairs[:64]):
    dyv[j] = a; wv[j] = b
dyh = dyv.half(); wh = wv.half()
T = tuple(t.to(dev) for t in (x, wh, dyh, rs))
dx, dw = kernel.run(*T)
rat = kernel.error_ratios((dx, dw), T)
ref_dx, _ = kernel.reference(*T)
max_dx_abs = float((dx.double() - ref_dx).abs().max())
# per-column exact-tie check that torch RN (ties-to-even, up here) differs from truncation
rn_val = torch.tensor((1.0 + 2 ** -10) * 1.5, dtype=torch.float16).item()
out["partC"] = dict(tie_pairs_used=len(pairs[:64]), torch_rn_product=rn_val,
                    trunc_product=math.nextafter(rn_val, 0.0) if rn_val > 0 else None,
                    dx_ratio=rat["dx"], dw_ratio=rat["dw"], max_dx_abs_diff=max_dx_abs,
                    dx_finite=bool(dx.isfinite().all()))
print(json.dumps(out))