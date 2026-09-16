
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

nchunks, dim = 8, 64
g = torch.Generator(device=dev).manual_seed(7)
ns = torch.randn(nchunks, dim, device=dev, generator=g, dtype=torch.float32)
# decays chosen so each sub-decay is exactly representable and sums exactly: total = -0.5*k_max multiples
base = torch.full((nchunks,), -1.0, device=dev, dtype=torch.float32)  # will scale per k
res = {}
for k in [1,2,3,4,8]:
    # per-chunk total decay -1.0 ; sub decays each -1.0/k -> for exactness use k in {1,2,4,8}; k=3 inexact on purpose
    sub = (base[0] / k)
    ns_s = torch.zeros(nchunks * k, dim, device=dev, dtype=torch.float32)
    da_s = torch.full((nchunks * k,), float(sub), device=dev, dtype=torch.float32)
    for c in range(nchunks):
        ns_s[c*k + (k-1)] = ns[c]
    out_split = m.state_passing(ns_s, da_s).double()
    da_un = base.clone()
    out_unsplit = m.state_passing(ns, da_un).double()
    r = ref_spec(ns, da_un)
    def rl(a,b):
        d=(a-b).abs(); return float((d/(b.abs()+1e-30)).max()), float(d.max())
    rs, as_ = rl(out_split, r)
    ru, au = rl(out_unsplit, r)
    sp_vs_un, sp_vs_un_abs = rl(out_split, out_unsplit)
    res[f"k{k}"] = {"split_rel_vs_fp64ref": rs, "unsplit_rel_vs_fp64ref": ru,
                    "split_vs_unsplit_rel": sp_vs_un, "split_vs_unsplit_abs": sp_vs_un_abs,
                    "sub_decay": float(sub), "sum_sub_decay": float(sub*k),
                    "nsubchunks": nchunks*k}

# also randomized decays split into unequal parts summing exactly
g2 = torch.Generator(device=dev).manual_seed(11)
da_r = -torch.nn.functional.softplus(torch.randn(nchunks, device=dev, generator=g2, dtype=torch.float32))
k = 4
ns_s = torch.zeros(nchunks*k, dim, device=dev, dtype=torch.float32)
da_s = torch.zeros(nchunks*k, device=dev, dtype=torch.float32)
for c in range(nchunks):
    parts = (da_r[c] / k)
    da_s[c*k:(c+1)*k] = parts
    ns_s[c*k + (k-1)] = ns[c]
out_split = m.state_passing(ns_s, da_s).double()
out_un = m.state_passing(ns, da_r).double()
r = ref_spec(ns, da_r)
d1 = (out_split-r).abs(); d2=(out_un-r).abs(); d3=(out_split-out_un).abs()
res["random_decays_k4"] = {"split_rel_vs_fp64ref": float((d1/(r.abs()+1e-30)).max()),
                           "unsplit_rel_vs_fp64ref": float((d2/(r.abs()+1e-30)).max()),
                           "split_vs_unsplit_rel": float((d3/(out_un.abs()+1e-30)).max())}
print(json.dumps({"per_k": res,
                  "metric": "relative error split-vs-unsplit and each vs common fp64 reference",
                  "benign_threshold": 1e-6, "structural_threshold": 1e-3}))
