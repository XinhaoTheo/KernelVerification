
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
    assert torch.isfinite(num).all() and torch.isfinite(den).all()
    out = k.run(num, den)
    tgt = fp64_ref(num, den)
    tol = 1e-7 + 5e-7*torch.abs(tgt)
    err = (out.to(torch.float64) - tgt).abs()
    viol = (err > tol)
    r = {"n": num.shape[0], "violations": int(viol.sum()),
         "max_err": float(err.max()), "max_ratio_err_over_tol": float((err/tol).max()),
         "finite": bool(torch.isfinite(out).all())}
    if viol.any():
        i = int((err/tol).argmax())
        r["worst_row"] = [float(num[i,0]), float(num[i,1]), float(den[i,0]), float(den[i,1]), float(out[i,0]), float(out[i,1]), float(tgt[i,0]), float(tgt[i,1])]
    results[label] = r
    return results

def rand_fp16(n, g, exp_lo=-24, exp_hi=14):
    e = torch.empty(n,2,dtype=torch.float64).uniform_(exp_lo, exp_hi, generator=g)
    m = torch.empty(n,2,dtype=torch.float64).uniform_(-1,1, generator=g)
    t = (m * 10.0**e).to(torch.float16)
    t[~torch.isfinite(t)] = 0.0
    return t.to(dev)

results = {}
num, den = k.make_inputs(dev)
results = check(num, den, "random_4093", results)
g = torch.Generator().manual_seed(1)
n = 131072
for trial in range(3):
    results = check(rand_fp16(n,g), rand_fp16(n,g), f"scaled_full_range_{trial}", results)
# cancellation: numerator = i*conj(denominator) -> real numerator ~ 0
c_ = torch.empty(n, dtype=torch.float64).uniform_(-1,1, generator=g)
d_ = torch.empty(n, dtype=torch.float64).uniform_(-1,1, generator=g)
den2 = torch.stack([c_, d_],1).to(torch.float16).to(dev)
num2 = torch.stack([d_, -c_],1).to(torch.float16).to(dev)
results = check(num2, den2, "cancelled_numerator", results)
# also extremes: numerator components at 65504 with opposite-sign products summing to near cancellation
e_ = torch.empty(n, dtype=torch.float64).uniform_(-1,1, generator=g)
num3 = torch.stack([65504.0*torch.ones(n), (65504.0*e_).to(torch.float64)],1).to(torch.float16).to(dev)
den3 = torch.stack([(65504.0*e_).to(torch.float64), -65504.0*torch.ones(n)],1).to(torch.float16).to(dev)
results = check(num3, den3, "extreme_cancellation", results)
# grids with subnormal / zero / extremal values
vals = [0.0, -0.0, 6e-8, 5.96e-8, 2.98e-8, 6.1e-5, 1.0, -1.0, 65504.0, -65504.0]
combos = list(itertools.product(vals, vals))
for ca in [0.0, 5.96e-8, 6.1e-5, 65504.0, 1.0]:
    pairs = [[v, ca] for v in vals] + [[ca, v] for v in vals]
    deng = torch.tensor(pairs, dtype=torch.float16, device=dev)
    numg = torch.tensor([[1.5, 2.5]]*len(pairs), dtype=torch.float16, device=dev)
    keep = (deng != 0).any(1)
    results = check(numg[keep], deng[keep], f"grid_c_{ca}", results)
# full all-pairs grid of extreme values for numerator too
allpairs = torch.tensor(combos, dtype=torch.float16, device=dev)
allpairs_f64 = allpairs.to(torch.float64)
denok = (allpairs != 0).any(1)
for base in [0.0, 5.96e-8, 6.1e-5, 65504.0, 1.0, 2048.0]:
    numg = torch.tensor([[base, base]]*len(combos), dtype=torch.float16, device=dev)
    results = check(numg[denok], allpairs[denok], f"allpairs_num_{base}", results)
# subnormal/big mixes
sub = torch.empty(n,2, dtype=torch.float64).uniform_(-6e-8, 6e-8, generator=g).to(torch.float16).to(dev)
big = rand_fp16(n, g, exp_lo=11, exp_hi=14)
results = check(big, sub, "big_over_subnormal", results)
results = check(sub, big, "subnormal_over_big", results)
results = check(sub.clone(), sub.clone(), "subnormal_over_subnormal", results)
# denominator with one zero component
z = torch.zeros(100000, dtype=torch.float16, device=dev)
o = torch.ones(100000, dtype=torch.float16, device=dev)
rn = torch.randn(100000,2, dtype=torch.float16, device=dev)
results = check(rn, torch.stack([o, z],1), "den_real_axis", results)
results = check(rn, torch.stack([z, o],1), "den_imag_axis", results)
# input preservation + freshness
num4, den4 = k.make_inputs(dev, size=777, seed=3)
n0, d0 = num4.clone(), den4.clone()
out4a = k.run(num4, den4); out4b = k.run(num4, den4)
results["inputs_preserved"] = bool(torch.equal(num4, n0) and torch.equal(den4, d0))
results["fresh_tensor"] = bool(out4a.data_ptr() != out4b.data_ptr() and out4a.dtype == torch.float32 and tuple(out4a.shape) == (777,2))
print(json.dumps(results, indent=1))
