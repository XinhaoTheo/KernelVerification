
import importlib.util, json, torch
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_02/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = "cuda"

def ref(ns, da):
    ns64, da64 = ns.double(), da.double()
    s = torch.zeros(ns.shape[1], dtype=torch.float64, device=ns.device)
    for c in range(ns.shape[0]):
        s = torch.exp(da64[c]) * s + ns64[c]
    return s

res = {}
for k in [2, 8, 16, 32]:
    nchunks, dim = 16, 64
    g = torch.Generator(device=dev).manual_seed(100 + k)
    ns = torch.randn(nchunks, dim, device=dev, generator=g, dtype=torch.float32)
    # unequal random sub-decays; unsplit decay defined as their fp32 sum so the partition sums match by construction
    parts = -torch.nn.functional.softplus(torch.randn(nchunks, k, device=dev, generator=g, dtype=torch.float32)) / k
    da_un = parts.sum(dim=1).contiguous()
    ns_s = torch.zeros(nchunks * k, dim, device=dev, dtype=torch.float32)
    da_s = parts.reshape(-1).contiguous()
    for c in range(nchunks):
        ns_s[c * k + (k - 1)] = ns[c]
    o_s = m.state_passing(ns_s, da_s).double()
    o_u = m.state_passing(ns, da_un).double()
    r = ref(ns, da_un)
    scale = float(r.abs().max())
    wc = r.abs() > 1e-2 * scale
    def stats(a, b):
        d = (a - b).abs()
        return {"max_abs": float(d.max()),
                "max_rel_all": float((d / (b.abs() + 1e-30)).max()),
                "max_rel_wellcond": float((d[wc] / b.abs()[wc]).max())}
    res[f"k{k}"] = {"nsubchunks": nchunks * k, "ref_absmax": scale,
                    "total_decay_min": float(da_un.min()), "total_decay_max": float(da_un.max()),
                    "split_vs_ref": stats(o_s, r), "unsplit_vs_ref": stats(o_u, r),
                    "split_vs_unsplit": stats(o_s, o_u)}
print(json.dumps({"per_k": res,
                  "metric": "split-vs-unsplit and each-vs-fp64ref: max abs, max rel all, max rel on well-conditioned elements",
                  "structural_threshold": 1e-3}))
