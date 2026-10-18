import torch, json, sys
sys.path.insert(0, "/root/cases/case_154")
import kernel as K

dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(7)

def rand_fp16(n, emin=-3, emax=15):
    e = torch.empty(n).uniform_(emin, emax, generator=g)
    mant = 1.0 + torch.empty(n).uniform_(0, 1, generator=g)
    s = torch.where(torch.empty(n).uniform_(-1, 1, generator=g) < 0, -1.0, 1.0)
    return (s * mant * (2.0 ** e)).to(torch.float16)

fails_total = 0
worst_ratio = 0.0
examples = []
rows_tested = 0
for trial in range(40):
    n = 65536
    a = rand_fp16(n); b = rand_fp16(n); c = rand_fp16(n)
    half = n // 2
    # real-part near cancellation: d ~ -a*c/b (clipped to finite fp16)
    d_real = (-(a[:half].float() * c[:half].float()) / b[:half].float().clamp(min=1e-30)).to(torch.float16)
    # imag-part near cancellation: d ~ b*c/a
    d_imag = ((b[half:].float() * c[half:].float()) / a[half:].float().clamp(min=1e-30)).to(torch.float16)
    d = torch.cat([d_real, d_imag]).to(torch.float16)
    # perturb half the rows by 1 fp16 ulp
    bits = d.view(torch.int16)
    pert = torch.where(torch.empty(n).uniform_(-1, 1, generator=g) < 0, bits + 1, bits)
    d = pert.to(torch.int16).view(torch.float16)
    keep = torch.isfinite(d) & (torch.abs(d) <= 65504) & (a != 0) & (b != 0) & (c != 0)
    d = d * keep  # zero out invalid (den row nonzero requires c!=0 kept, c kept nonzero)
    num = torch.stack([a, b], dim=1)[keep].to(dev).to(torch.float16)
    den = torch.stack([c, d], dim=1)[keep].to(dev).to(torch.float16)
    if den.shape[0] == 0:
        continue
    assert num.dtype == torch.float16 and den.dtype == torch.float16
    assert bool(torch.isfinite(num).all()) and bool(torch.isfinite(den).all())
    rows_tested += den.shape[0]
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
            examples.append({"num_row": num.cpu()[i].tolist(), "den_row": den.cpu()[i].tolist(),
                             "out": out.cpu()[i].tolist(), "ref": ref.cpu()[i].tolist(),
                             "err": float(err.cpu()[i, j]), "tol": float(tol.cpu()[i, j]),
                             "ratio": float(ratio.cpu()[i, j]), "comp": j})
print(json.dumps({"trials": 40, "rows_tested": rows_tested, "fails_total": fails_total,
                  "worst_ratio": worst_ratio, "examples": examples[:5]}))
