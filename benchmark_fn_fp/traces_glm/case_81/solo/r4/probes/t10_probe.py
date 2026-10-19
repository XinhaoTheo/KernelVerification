import torch, json, itertools
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"

def fp64_ref(num, den):
    a, b = num.to(torch.float64).unbind(1); c, d = den.to(torch.float64).unbind(1)
    s = c*c + d*d
    return torch.stack(((a*c+b*d)/s, (b*c-a*d)/s), 1)

def check(num, den, label, results):
    num = num.to(dev); den = den.to(dev)
    keep = (den != 0).any(1) & torch.isfinite(num).all(1) & torch.isfinite(den).all(1)
    num, den = num[keep], den[keep]
    if num.shape[0] == 0:
        results[label] = {"skipped": True}; return results
    out = k.run(num, den)
    tgt = fp64_ref(num, den)
    tol = 1e-7 + 5e-7*torch.abs(tgt)
    err = (out.to(torch.float64) - tgt).abs()
    viol = (err > tol)
    r = {"n": num.shape[0], "violations": int(viol.sum()),
         "max_ratio_err_over_tol": float((err/tol).max()),
         "finite": bool(torch.isfinite(out).all())}
    if viol.any():
        i = int((err/tol).argmax())
        r["worst_row"] = [float(num[i,0]), float(num[i,1]), float(den[i,0]), float(den[i,1]), float(out[i,0]), float(out[i,1]), float(tgt[i,0]), float(tgt[i,1]), float(err[i]/tol[i])]
    results[label] = r
    return results

results = {}
num, den = k.make_inputs(dev)
results = check(num, den, "random_4093", results)

g = torch.Generator().manual_seed(1)
n = 131072
def rand_fp16(n, g, exp_lo=-24, exp_hi=14):
    e = torch.empty(n,2,dtype=torch.float64).uniform_(exp_lo, exp_hi, generator=g)
    m = torch.empty(n,2,dtype=torch.float64).uniform_(-1,1, generator=g)
    t = (m * 10.0**e).to(torch.float16)
    t[~torch.isfinite(t)] = 0.0
    return t.to(dev)

for trial in range(3):
    results = check(rand_fp16(n,g), rand_fp16(n,g), f"scaled_full_range_{trial}", results)

c_ = torch.empty(n, dtype=torch.float64).uniform_(-1,1, generator=g)
d_ = torch.empty(n, dtype=torch.float64).uniform_(-1,1, generator=g)
den2 = torch.stack([c_, d_],1).to(torch.float16).to(dev)
num2 = torch.stack([d_, -c_],1).to(torch.float16).to(dev)
results = check(num2, den2, "cancelled_numerator", results)

e_ = torch.empty(n, dtype=torch.float64).uniform_(-1,1, generator=g)
num3 = torch.stack([65504.0*torch.ones(n), (65504.0*e_)],1).to(torch.float16).to(dev)
den3 = torch.stack([(65504.0*e_), -65504.0*torch.ones(n)],1).to(torch.float16).to(dev)
results = check(num3, den3, "extreme_cancellation", results)

vals = [0.0, -0.0, 5.96e-8, 2.98e-8, 6.1e-5, 1.0, -1.0, 65504.0, -65504.0]
combos = list(itertools.product(vals, vals))
deng = torch.tensor(combos, dtype=torch.float16)
for base in [5.96e-8, 6.1e-5, 1.0, 2048.0, 65504.0]:
    numg = torch.tensor([[base, base]]*len(combos), dtype=torch.float16)
    results = check(numg, deng, f"allpairs_den_{base}", results)
small = torch.tensor(combos, dtype=torch.float16)
nn = small.repeat(len(combos),1); dd = small.tile((len(combos),1))
results = check(nn, dd, "allpairs_x_allpairs", results)

sub = torch.empty(n,2, dtype=torch.float64).uniform_(-6e-8, 6e-8, generator=g).to(torch.float16).to(dev)
big = rand_fp16(n, g, exp_lo=11, exp_hi=14)
results = check(big, sub, "big_over_subnormal", results)
results = check(sub, big, "subnormal_over_big", results)
results = check(sub.clone(), sub.clone(), "subnormal_over_subnormal", results)

z = torch.zeros(100000, dtype=torch.float16); o = torch.ones(100000, dtype=torch.float16)
rn = torch.randn(100000,2, dtype=torch.float16)
results = check(rn, torch.stack([o, z],1), "den_real_axis", results)
results = check(rn, torch.stack([z, o],1), "den_imag_axis", results)

num4, den4 = k.make_inputs(dev, size=777, seed=3)
n0, d0 = num4.clone(), den4.clone()
out4 = k.run(num4, den4)
results["inputs_preserved"] = bool(torch.equal(num4, n0) and torch.equal(den4, d0))
results["output_shape_dtype_fresh"] = (tuple(out4.shape), str(out4.dtype), str(out4.device))

print(json.dumps(results, indent=1))
