
import torch, json, itertools, random
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_154/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
dev = "cuda"

def fp64_ref(num, den):
    a, b = num.to(torch.float64).unbind(1); c, d = den.to(torch.float64).unbind(1)
    s = c*c + d*d
    return torch.stack(((a*c+b*d)/s, (b*c-a*d)/s), 1)

def check(num, den, label, results):
    out = k.run(num, den)
    tgt = fp64_ref(num, den)
    tol = 1e-7 + 5e-7*torch.abs(tgt)
    err = (out.to(torch.float64) - tgt).abs()
    viol = (err > tol)
    results[label] = {"n": num.shape[0], "violations": int(viol.sum()),
                      "max_err": float(err.max()), "max_tol": float(tol.max()),
                      "finite": bool(torch.isfinite(out).all())}
    if viol.any():
        i = int(err.div(tol).argmax())
        results[label]["worst_row"] = [float(num[i,0]), float(num[i,1]), float(den[i,0]), float(den[i,1]), float(out[i,0]), float(out[i,1]), float(tgt[i,0]), float(tgt[i,1])]
    return results

results = {}
# 1. random
num, den = k.make_inputs(dev)
results = check(num, den, "random_4093", results)
# 2. random extrema-scaled magnitudes: random exponents over full fp16 range
g = torch.Generator().manual_seed(1)
n = 200000
for trial in range(3):
    def sampler():
        e = torch.empty(n,2,dtype=torch.float64).uniform_(-24, 15, generator=g)
        m = torch.empty(n,2,dtype=torch.float64).uniform_(-1,1, generator=g)
        return (m * 10.0**e).to(torch.float16).to(dev)
    results = check(sampler(), sampler(), f"scaled_full_range_{trial}", results)
# 3. crafted cancellation: numerator ~ proportional to denominator conjugate => tiny real part
# choose a,b such that a*c + b*d is tiny relative to magnitudes
c_ = torch.empty(n, dtype=torch.float64).uniform_(-1,1, generator=g)
d_ = torch.empty(n, dtype=torch.float64).uniform_(-1,1, generator=g)
a_ = d_; b_ = -c_  # numerator = i*conj? a*c+b*d = d*c - c*d = 0 exactly at fp64; quantize to fp16
den2 = torch.stack([c_, d_],1).to(torch.float16).to(dev)
num2 = torch.stack([a_, b_],1).to(torch.float16).to(dev)
results = check(num2, den2, "cancelled_numerator", results)
# 4. subnormals / zeros / extrema grid
vals = [0.0, -0.0, 5e-324, 6e-8, 6.1e-5, 1.0, -1.0, 65504.0, -65504.0, 5.96e-8, 2.98e-8]
combos = list(itertools.product(vals, vals))
for ca in [0.0, 6.1e-5, 65504.0, 5.96e-8, 1.0]:
    pairs = [[v, ca] for v in vals] + [[ca, v] for v in vals]
    numg = torch.tensor([[1.5, 2.5]]*len(pairs), dtype=torch.float16, device=dev)
    deng = torch.tensor(pairs, dtype=torch.float16, device=dev)
    results = check(numg, deng, f"grid_c_{ca}", results)
# 5. brute random subnormal-heavy
sub = torch.empty(200000,2, dtype=torch.float64).uniform_(-6e-8, 6e-8, generator=g).to(torch.float16).to(dev)
big = (torch.empty(200000,2, dtype=torch.float64).uniform_(-65504, 65504, generator=g)).to(torch.float16).to(dev)
results = check(big, sub, "big_over_subnormal", results)
results = check(sub, big, "subnormal_over_big", results)
results = check(sub.clone(), sub.clone(), "subnormal_over_subnormal", results)
# 6. small denominator rows with one zero component
z = torch.zeros(100000, dtype=torch.float16, device=dev)
o = torch.ones(100000, dtype=torch.float16, device=dev)
rn = torch.randn(100000,2, dtype=torch.float16, device=dev)
results = check(rn, torch.stack([o, z],1), "den_real_axis", results)
results = check(rn, torch.stack([z, o],1), "den_imag_axis", results)
# also check inputs preserved
num3, den3 = k.make_inputs(dev, size=777, seed=3)
n0, d0 = num3.clone(), den3.clone()
k.run(num3, den3)
results["inputs_preserved"] = bool(torch.equal(num3, n0) and torch.equal(den3, d0))
print(json.dumps(results, indent=1))
