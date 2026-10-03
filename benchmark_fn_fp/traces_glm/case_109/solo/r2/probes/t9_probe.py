
import torch, json, sys, os
sys.path.insert(0, "/root/cases/case_109")
import kernel as K

dev = "cuda"
def contract(x, weight, dy, rstd):
    # independent FP64 implementation of problem.txt formulas
    x64 = x.double(); w64 = weight.double(); d64 = dy.double(); r64 = rstd.double()[:, None]
    m = (dy.float() * weight.float()).half().double()          # H16(dy*w)
    h = (x.float() * rstd[:, None]).half().double()            # H16(F32(F32(x)*rstd))
    t = (dy.float() * h.half().float()).half().double()        # H16(dy*h)
    dx = r64 * (m - x64 * r64.square() * (m * x64).sum(1, keepdim=True) / x.shape[1])
    dw = t.sum(0)
    return dx, dw, t

def check(m, n, seed, extreme=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if extreme:
        # push values to domain extremes, rows with RMS near 0.25 and 4
        x = (torch.empty(m, n).uniform_(-4, 4, generator=g)).half()
        # scale rows so RMS hits ~0.25 or ~4
        for i in range(m):
            rms = x[i].float().square().mean().sqrt()
            target = 4.0 if i % 2 else 0.25
            x[i] = (x[i].float() * (target / (rms + 1e-9))).clamp(-4, 4).half()
        # rescale slightly below 4 to keep |x|<=4 after rounding
        x = (x.float() * 0.99).half()
        weight = (2.0 * torch.ones(n)).half()
        dy = (torch.empty(m, n).uniform_(-4, 4, generator=g)).half()
    else:
        x = (torch.empty(m, n).uniform_(-4, 4, generator=g)).half()
        weight = (torch.empty(n).uniform_(0.05, 2.0, generator=g)).half()
        dy = (torch.empty(m, n).uniform_(-4, 4, generator=g)).half()
    # verify row RMS in [0.25,4]; rescale otherwise
    rms = x.float().square().mean(1).sqrt()
    lo, hi = rms.min().item(), rms.max().item()
    if lo < 0.24 or hi > 4.01:
        # scale rows into band
        sc = torch.where(rms < 0.25, 0.25/(rms+1e-9), torch.where(rms > 4.0, 4.0/(rms+1e-9), torch.ones_like(rms)))
        x = (x.float() * sc[:, None]).clamp(-4, 4).half()
        rms = x.float().square().mean(1).sqrt()
        lo, hi = rms.min().item(), rms.max().item()
    eps = 1e-5
    rstd = (1.0 / torch.sqrt(x.float().square().mean(1) + eps)).float()
    x0, w0, d0, r0 = (t.clone() for t in (x, weight, dy, rstd))
    x, weight, dy, rstd = (t.to(dev) for t in (x, weight, dy, rstd))
    dx, dw = K.run(x, weight, dy, rstd)
    inputs = (x, weight, dy, rstd)
    rdx, rdw, t = contract(x, weight, dy, rstd)
    # error bounds
    dx_lim = 0.002 + 0.002 * rdx.abs()
    dw_lim = 1e-5 + 1e-5 * t.abs().sum(0)
    dxr = ((dx.double() - rdx).abs() / dx_lim)
    dwr = ((dw.double() - rdw).abs() / dw_lim)
    # also use kernel's own error_ratios
    ratios = K.error_ratios((dx, dw), inputs)
    mutated = any(not torch.equal(a.cpu(), b) for a, b in zip((x, weight, dy, rstd), (x0, w0, d0, r0)))
    finite = torch.isfinite(dx.float()).all().item() and torch.isfinite(dw).all().item()
    return dict(m=m, n=n, rms_band=(round(lo,4), round(hi,4)),
                dx_ratio=float(dxr.max().item()), dw_ratio=float(dwr.max().item()),
                kernel_error_ratios=ratios, finite=bool(finite), mutated=mutated)

results = []
# shapes: exercise block kernel (block<=256, m>=512) and row kernel (block>256 i.e. n>256, or m<512)
for (m, n, ex) in [(512,128,False),(512,64,False),(768,128,False),(4096,512,False),(4096,256,False),
                   (512,128,True),(4096,512,True),
                   (511,100,False),(513,300,False),(4095,511,False),(1,16,False),(17,31,False),
                   (127,257,False),(1000,129,False),(2048,100,False),(300,400,False),(2,512,False)]:
    results.append(check(m, n, seed=m*7+n, extreme=ex))

print(json.dumps(results, indent=1))
worst_dx = max(r["dx_ratio"] for r in results)
worst_dw = max(r["dw_ratio"] for r in results)
print(json.dumps({"worst_dx_ratio": worst_dx, "worst_dw_ratio": worst_dw,
                  "all_finite": all(r["finite"] for r in results),
                  "any_mutation": any(r["mutated"] for r in results)}))
