
import torch, sys, json, os
sys.path.insert(0, "/root/cases/case_154")
import kernel

dev = "cuda"
torch.manual_seed(0)

rows = []
# Construct cancellation: real numerator a*c + b*d ~ 0 exactly or tiny
# pick a*c = b*d (exact in fp16) => real numerator exactly 0 (harmless)
# better: a*c + b*d small but products huge relative => rounding error
# e.g. a=1e4, c=1e4 product 1e8; b,d chosen so b*d = -(a*c - eps). But fp16 quantizes eps.
# Use imag numerator b*c - a*d with b*c and a*d nearly equal.
def add(a,b,c,d):
    rows.append((a,b,c,d))

# systematic cancellation: for fp16 values, b*c - a*d exactly representable? Try a/b ratio approximating c... 
# construct: let u,v be fp16. set a=u, d=v, b=u, c=v -> imag num = u*v - u*v =0 exact.
# Instead: b = u, c = v1; a = u, d = v2 with v1,v2 adjacent fp16 values: num = u*(v1-v2)
# v1-v2 = 1 ulp of v. num = u*ulp(v). kernel fp32: u*v1 exact, u*v2 exact, subtraction exact (both fp32 representable, difference maybe subnormal fp32? u*ulp(v) exact in fp32? u<=65504, ulp(v)<=64 => product <= ~4.2e6, representable). Actually both products exact, diff of two exact fp32 numbers is exact if representable. u*v1 - u*v2 = u*(v1-v2); |u*(v1-v2)| may be subnormal? no, it's >= u*min_ulp. Exact if within fp32 precision: difference of two fp32s is exactly representable when close (Sterbenz) or small; generally fp32 has 24 bits; products need <=11 bits each; difference needs <= 22 bits -> exact. So this path is exact.
# The real error source: division rounding (div_rn, 0.5 ulp) and scale computation c*c+d*d rounding.
# Cancellation case where fp32 sum of products is NOT exact: a*c + b*d with opposite signs and different exponents -> rounding to 24 bits of a value that needs up to 22 bits -> exact actually. Hmm products are each <= 2^22-ish in ulps... products of two 11-bit significands need 22 bits; sum of two needs up to 23 bits (or more with exponent spread). Exponent spread -> rounding.
# Classic failing case: real num small relative to terms with exponent spread, e.g. a*c = 65504*65504 ~ 4.29e9 (needs 2 bits + exponent) and b*d = -(4.29e9 - tiny). tiny must be fp16-quantizable difference of products... 
# Simplest robust approach: random search over all fp16 values for max violation.

# Build a large random search over fp16 values in a reasonable range plus systematic edge values
import itertools, random
random.seed(0)

vals = set()
# fp16 positive values
x = torch.tensor([0.0])
for e in range(-14, 16):
    for m in range(1024):
        pass
# easier: generate fp16 grid pieces
grids = []
for scale_exp in range(-14, 16):
    base = 2.0**scale_exp
    mant = torch.arange(1024, dtype=torch.float32)/1024.0
    g = (1+mant)*base
    grids.append(g)
pos = torch.cat(grids)  # all positive normal fp16
pos = pos.to(torch.float16)
extra = torch.tensor([0.0, 5.96e-8, 6.1e-5, 1e-3, 1, 2, 4, 100, 1000, 65504, 6.0976e-5, 5.9605e-8], dtype=torch.float16)
pos_all = torch.unique(torch.cat([pos, extra.to(torch.float32).to(torch.float16)]))
neg_all = torch.cat([-pos_all, pos_all])

worst = None
worst_info = None
count_viol = 0
trials = 0

def check(num, den):
    global worst, worst_info, count_viol, trials
    n = num.shape[0]
    out = kernel.run(num, den)
    ref = kernel.reference(num.cpu(), den.cpu()).to(dev)
    err = (out - ref).abs()
    tol = 1e-7 + 5e-7*ref.abs()
    viol = err > tol
    if viol.any():
        cnt = int(viol.sum())
        count_viol += cnt
        ratio = ((err - tol)/tol)[viol].max().item()
        if worst is None or ratio > worst:
            worst = ratio
            idx = int(viol.nonzero()[0,0])
            worst_info = dict(row=idx, num=num[idx].tolist(), den=den[idx].tolist(),
                              out=out[idx].tolist(), ref=ref[idx].tolist(),
                              err=err[idx].tolist(), tol=tol[idx].tolist())
    trials += n
    return int(viol.sum())

# random fp16 combos, bias to cancellation-prone
for it in range(300):
    n = 4096
    # sample from pos_all & negatives, biased toward similar magnitudes in numerator vs denominator rows
    a = pos_all[torch.randint(0, len(pos_all), (n,))].to(dev) * (torch.randint(0,2,(n,))*2-1).float()
    b = pos_all[torch.randint(0, len(pos_all), (n,))].to(dev) * (torch.randint(0,2,(n,))*2-1).float()
    c = pos_all[torch.randint(0, len(pos_all), (n,))].to(dev) * (torch.randint(0,2,(n,))*2-1).float()
    d = pos_all[torch.randint(0, len(pos_all), (n,))].to(dev) * (torch.randint(0,2,(n,))*2-1).float()
    num = torch.stack([a,b],1).to(torch.float16)
    den = torch.stack([c,d],1).to(torch.float16)
    # ensure denominator nonzero rows
    z = (den==0).all(dim=1)
    den[z,0] = 1
    check(num, den)

# targeted: a=c and b=-d? then real num = a*c + b*d = a^2 - b^2 fine. 
# targeted cancellation: make b*c ≈ a*d with big exponent spread: a=2^-14 subnormal u, b=huge B... 
# numerator row proportional: (a,b) = t*(d,-c)? then real num = a*c+b*d = t*(d*c - c*d)=0 exact.
# offset by one ulp: a = t*d, b = -t*c adjusted... use imag: b*c - a*d.
# direct: set b = w, c = P (big), a = w, d = P2 where P2 = nextafter(P) in fp16 -> imag num = w*(P-P2) tiny while terms w*P huge.
for w_exp in range(-14, 16):
    for p_exp in range(-5, 16):
        w = (2.0**w_exp)
        P = (2.0**p_exp)*(1+ 512/1024.0)  # has ulp 2^(p_exp-10)
        P2 = P + 2.0**(p_exp-10)
        n = 1
        num = torch.tensor([[w, w]], dtype=torch.float16, device=dev)
        den = torch.tensor([[P2, P]], dtype=torch.float16, device=dev)
        check(num, den)
        # real numerator analog
        num2 = torch.tensor([[w, -w]], dtype=torch.float16, device=dev)
        den2 = torch.tensor([[P, -P2]], dtype=torch.float16, device=dev)
        check(num2, den2)

print(json.dumps(dict(trials=trials, violations=count_viol, worst_excess_ratio=worst, worst_info=worst_info)))
