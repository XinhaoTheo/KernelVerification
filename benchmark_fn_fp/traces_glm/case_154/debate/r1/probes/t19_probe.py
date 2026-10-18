import torch, json, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel as K

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(99)
n = 131072
fails_total = 0
worst_ratio = 0.0
worst_info = None
for trial in range(60):
    e = torch.empty(n, 4).uniform_(-5, 15, generator=g)
    mant = 1.0 + torch.empty(n, 4).uniform_(0, 1, generator=g)
    sgn = torch.where(torch.empty(n, 4).uniform_(-1, 1, generator=g) < 0, -1.0, 1.0)
    x = (sgn * mant * (2.0 ** e)).to(torch.float16)
    a, b, c, d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    if trial % 2 == 0:
        # no-cancellation for real part: make b*d same sign as a*c
        b = b.abs() * torch.where(a < 0, -1.0, 1.0)
        d = d.abs() * torch.where(c < 0, -1.0, 1.0)
    if trial % 3 == 0:
        # skewed denominator: |d| ~ |c| * 2^-14 so scale dominated by c^2
        d = (c.float() * (2.0 ** -14)).to(torch.float16)
    num = torch.stack([a, b], dim=1).to(torch.float16).to(dev)
    den = torch.stack([c, d], dim=1).to(torch.float16).to(dev)
    z = (den == 0).all(dim=1)
    den[z, 0] = torch.tensor(1.0, dtype=torch.float16, device=den.device)
    den = den.to(torch.float16)
    assert num.dtype == torch.float16 and den.dtype == torch.float16
    assert bool(torch.isfinite(num).all()) and bool(torch.isfinite(den).all())
    out = K.run(num, den)
    ref = K.reference(num, den)
    err = (out.double() - ref.double()).abs()
    tol = 1e-7 + 5e-7 * ref.abs()
    ratio = err / tol
    f = ratio > 1.0
    fails_total += int(f.sum())
    mr = float(ratio.max())
    if mr > worst_ratio:
        worst_ratio = mr
        i0, j0 = torch.unravel_index(ratio.argmax(), ratio.shape)
        worst_info = {"num_row": num.cpu()[int(i0)].tolist(),
                      "den_row": den.cpu()[int(i0)].tolist(),
                      "out": out.cpu()[int(i0)].tolist(),
                      "ref": ref.cpu()[int(i0)].tolist(),
                      "err": float(err.cpu()[int(i0), int(j0)]),
                      "tol": float(tol.cpu()[int(i0), int(j0)]),
                      "ratio": mr, "comp": int(j0)}
print(json.dumps({"trials": 60, "rows_per_trial": n, "fails_total": fails_total,
                  "worst_ratio": worst_ratio, "worst_case": worst_info}))
