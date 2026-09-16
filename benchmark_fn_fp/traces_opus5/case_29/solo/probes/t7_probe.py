
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_29/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

dev='cuda'
def ref(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)

# targeted subnormal-range values
vals = [0.001, 0.002, 0.005, 0.01, 0.0146, 0.015, 0.0001, -0.003, 0.0019, 0.0009765625]
x = torch.tensor(vals, dtype=torch.float32, device=dev)
out = m.fp8_roundtrip(x).cpu()
r = ref(x).cpu()
targeted = [{"x":v,"kernel":float(o),"ref_e4m3":float(rr),"abs_err":abs(float(o)-float(rr))}
            for v,o,rr in zip(vals,out,r)]

# representability check: does kernel output round-trip through e4m3 unchanged?
torch.manual_seed(0)
g = torch.randn(200000, device=dev)
og = m.fp8_roundtrip(g)
rg = ref(g)
representable = torch.isclose(og, ref(og)).float().mean().item()
mismatch = (og != rg)
n_mis = int(mismatch.sum())
# among small |x|
small = g.abs() < 2**-6
n_small = int(small.sum())
n_mis_small = int((mismatch & small).sum())
n_mis_large = int((mismatch & ~small).sum())
maxrel = ((og-rg).abs()/rg.abs().clamp_min(1e-30))[mismatch].max().item() if n_mis else 0.0

print(json.dumps({
 "metric":"exact match vs torch float8_e4m3fn quantization + representability of kernel output",
 "targeted":targeted,
 "randn_n":200000,"n_mismatch":n_mis,"frac_output_representable":representable,
 "n_small_lt_2^-6":n_small,"n_mismatch_small":n_mis_small,"n_mismatch_large":n_mis_large,
 "max_rel_err_on_mismatch":maxrel}, indent=1))
