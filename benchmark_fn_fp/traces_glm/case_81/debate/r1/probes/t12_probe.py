import torch, json, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel as K

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(7)

def rand_fp16(n, emin=-3, emax=15):
    e = torch.empty(n).uniform_(emin, emax, generator=g)
    mant = 1.0 + torch.empty(n).uniform_(0, 1, generator=g)
    s = torch.where(torch.empty(n).uniform_(-1, 1, generator=g) < 0, -1.0, 1.0)
    x = s * mant * (2.0 ** e)
    return x.to(torch.float16)

n = 131072
fails_total = 0
worst_ratio = 0.0
examples = []
for trial in range(40):
    a = rand_fp16(n); b = rand_fp16(n); c = rand_fp16(n)
    # real cancellation: a*c + b*d ~ 0 -> d = -a*c/b ; imag cancellation rows too
    half = n // 2
    d_real = (-(a[:half].float() * c[:half].float()) / b[:half].float()).to(torch.float16)
    d_imag = (( b[half:].float() * c[half:].float()) / a[half:].float()).to(torch.float16)
    d = torch.cat([d_real, d_imag])
    keep = torch.isfinite(d) & (b[:half].abs() > 0)[: :, ].expand(-1) if False else torch.ones(n, dtype=torch.bool)
    keep[:half] &= (b[:half] != 0) & (c[:half] != 0)
    keep[half:] &= (a[half:] != 0) & (c[half:] != 0)
    num = torch.stack([a, b], dim=1)[keep].to(dev)
    den = torch.stack([c, d], dim=1)[keep].to(dev)
    z = (den == 0).all(dim=1)
    den[z, 0] = 1.0
    out = K.run(num, den)
    ref = K.reference(num, den)
    err = (out.double() - ref.double()).abs()
    tol = 1e-7 + 5e-7 * ref.abs()
    ratio = err / tol
    f = ratio > 1.0
    nf = int(f.sum())
    fails_total += nf
    mr = float(ratio.max())
    if mr > worst_ratio:
        worst_ratio = mr
    if nf > 0 and len(examples) < 5:
        idx = torch.nonzero(f)
        for ij in idx[:5 - len(examples)]:
            i = int(ij[0]); j = int(ij[1])
            examples.append({
                "num_row": num.cpu()[i].tolist(), "den_row": den.cpu()[i].tolist(),
                "out": out.cpu()[i].tolist(), "ref": ref.cpu()[i].tolist(),
                "err": float(err.cpu()[i, j]), "tol": float(tol.cpu()[i, j]),
                "ratio": float(ratio.cpu()[i, j]), "comp": j})
print(json.dumps({"rows_tested": 40 * int(keep.numel()) if False else None,
                  "trials": 40, "fails_total": fails_total,
                  "worst_ratio": worst_ratio, "examples": examples[:5]}))
