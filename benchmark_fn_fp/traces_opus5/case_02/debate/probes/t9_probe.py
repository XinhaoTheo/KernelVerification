
import importlib.util, json, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_02/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

dev = "cuda"

def ref_spec(ns, da):
    ns64, da64 = ns.double(), da.double()
    s = torch.zeros(ns.shape[1], dtype=torch.float64, device=ns.device)
    for c in range(ns.shape[0]):
        s = torch.exp(da64[c]) * s + ns64[c]
    return s

def ref_decay_on_new(ns, da):  # wrong variant: decay applied to new_states
    ns64, da64 = ns.double(), da.double()
    s = torch.zeros(ns.shape[1], dtype=torch.float64, device=ns.device)
    for c in range(ns.shape[0]):
        s = s + torch.exp(da64[c]) * ns64[c]
    return s

def ref_shift(ns, da, sh):  # wrong variant: dA_cs index shifted
    ns64, da64 = ns.double(), da.double()
    n = ns.shape[0]
    s = torch.zeros(ns.shape[1], dtype=torch.float64, device=ns.device)
    for c in range(n):
        idx = min(max(c + sh, 0), n - 1)
        s = torch.exp(da64[idx]) * s + ns64[c]
    return s

def rel(a, b):
    d = (a - b).abs()
    return float((d / (b.abs() + 1e-30)).max()), float(d.max()), float(b.abs().max())

res = {}
for (nchunks, dim, seed) in [(8,64,0),(1,64,1),(16,128,2),(33,64,3),(64,256,4)]:
    g = torch.Generator(device=dev).manual_seed(seed)
    ns = torch.randn(nchunks, dim, device=dev, generator=g, dtype=torch.float32)
    da = -torch.nn.functional.softplus(torch.randn(nchunks, device=dev, generator=g, dtype=torch.float32))
    out = m.state_passing(ns, da).double()
    r = ref_spec(ns, da)
    mr, ma, scale = rel(out, r)
    res[f"n{nchunks}_d{dim}"] = {
        "max_rel_err_vs_spec": mr, "max_abs_err_vs_spec": ma, "ref_absmax": scale,
        "max_rel_vs_decay_on_new_variant": rel(out, ref_decay_on_new(ns, da))[0],
        "max_rel_vs_shift_plus1": rel(out, ref_shift(ns, da, 1))[0],
        "max_rel_vs_shift_minus1": rel(out, ref_shift(ns, da, -1))[0],
        "nonfinite": int((~torch.isfinite(out)).sum()),
    }

worst = max(v["max_rel_err_vs_spec"] for v in res.values())
print(json.dumps({"per_case": res, "worst_max_rel_err_vs_spec": worst,
                  "metric": "max elementwise relative error of returned final state vs fp64 sequential reference",
                  "threshold_structural": 1e-4}, indent=None))
