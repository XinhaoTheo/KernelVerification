
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
worst_tail = 0
worst_head = 0
for dim in [65, 100, 127, 129, 255, 257, 511, 1000, 1023]:
    for seed in [0, 1, 2]:
        nchunks = 12
        g = torch.Generator(device=dev).manual_seed(seed * 1000 + dim)
        ns = torch.randn(nchunks, dim, device=dev, generator=g, dtype=torch.float32)
        da = -torch.nn.functional.softplus(torch.randn(nchunks, device=dev, generator=g, dtype=torch.float32))
        o = m.state_passing(ns, da).double()
        r = ref(ns, da)
        scale = float(r.abs().max())
        aerr = (o - r).abs()
        rerr = aerr / (r.abs() + 1e-30)
        atol = 1e-5 * scale
        mism = aerr > atol
        p2 = 1 << (dim.bit_length() - 1)
        tail = torch.arange(dim, device=dev) >= p2
        wc = r.abs() > 1e-2 * scale
        i = int(rerr.argmax())
        mh = int((mism & ~tail).sum()); mt = int((mism & tail).sum())
        worst_head = max(worst_head, mh); worst_tail = max(worst_tail, mt)
        res[f"d{dim}_s{seed}"] = {
            "max_abs_err": float(aerr.max()), "ref_absmax": scale,
            "max_rel_err_all": float(rerr.max()),
            "max_rel_err_wellcond": float(rerr[wc].max()),
            "worst_rel_idx": i, "worst_rel_ref_value": float(r[i]), "worst_rel_abs_err": float(aerr[i]),
            "atol_used": atol,
            "mism_head_atol": mh, "mism_tail_atol": mt,
            "nonfinite": int((~torch.isfinite(o)).sum()),
        }
print(json.dumps({"per_case": res,
                  "worst_tail_mismatch_count_atol": worst_tail,
                  "worst_head_mismatch_count_atol": worst_head,
                  "metric": "abs-tolerance (1e-5*ref_absmax) mismatch counts split head/tail, plus rel err restricted to well-conditioned elements (|ref| > 1e-2*max|ref|)"}))
