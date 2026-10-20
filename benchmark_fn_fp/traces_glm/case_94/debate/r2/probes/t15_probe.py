import torch, json, sys, struct
sys.path.insert(0, "/root/cases/case_166")
import kernel as K

vals = []
for bits in range(0x0000, 0x10000):
    v = struct.unpack('<e', struct.pack('<H', bits))[0]
    if v == v and abs(v) != float('inf') and v > -1:
        vals.append(v)
x = torch.tensor(vals, dtype=torch.float16).to("cuda")
out = K.run(x)
ref = torch.log1p(x.to(torch.float64)).to("cuda")
err = (out.to(torch.float64) - ref).abs()
viol = (err > 1e-6)
vx = x[viol]; ve = err[viol]
u = (1.0 + vx.to(torch.float32)).to(torch.float64)
exps = torch.floor(torch.log2(u)).to(torch.int32).tolist()
# reduced mantissa m = u / 2^exponent (after kernel's >1.65 halving)
man = (u / (2.0 ** torch.tensor(exps, dtype=torch.float64, device=u.device))).tolist()
red_m = [m/2 if m > 1.65 else m for m in man]
# any violation with reduced mantissa NOT near 1.65 (say < 1.5) at large exponent?
far_from_cutoff = [abs(r-1.65) > 0.15 for r in red_m]
res = {
 "metric": "classify all domain-wide tolerance violations by exponent of u and reduced mantissa",
 "n_violations": int(viol.sum().item()),
 "violating_x": vx.tolist(),
 "violating_err": ve.tolist(),
 "violating_exponents": exps,
 "violating_reduced_mantissas": red_m,
 "n_violations_far_from_1.65_cutoff": int(sum(far_from_cutoff)),
 "n_violations_exponent_ge_13": sum(1 for e in exps if e >= 13),
 "n_violations_exponent_ge_13_far_from_cutoff": sum(1 for e, f in zip(exps, far_from_cutoff) if e >= 13 and f),
 "max_abs_err_large_exponent_any_mantissa": err[(torch.floor(torch.log2((1.0+x.to(torch.float32)).to(torch.float64))) >= 13)].max().item(),
 "tolerance": 1e-6,
}
print(json.dumps(res))
