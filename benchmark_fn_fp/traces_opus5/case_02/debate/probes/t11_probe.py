
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

res = {}
for dim in [1,3,65,100,127,257,1000]:
    nchunks = 12
    g = torch.Generator(device=dev).manual_seed(dim)
    ns = torch.randn(nchunks, dim, device=dev, generator=g, dtype=torch.float32)
    da = -torch.nn.functional.softplus(torch.randn(nchunks, device=dev, generator=g, dtype=torch.float32))
    # run twice to expose any dependence on uninitialized memory
    o1 = m.state_passing(ns, da).double()
    for _ in range(3):
        torch.empty(dim*64, device=dev, dtype=torch.float32).fill_(float('nan'))
    o2 = m.state_passing(ns, da).double()
    r = ref_spec(ns, da)
    prev_p2 = 1 << (dim.bit_length()-1)
    tail_idx = torch.arange(dim, device=dev) >= prev_p2
    relerr = (o1-r).abs()/(r.abs()+1e-30)
    mismatch = relerr > 1e-5
    res[f"dim{dim}"] = {
        "block_size": 1 << ((dim-1).bit_length()) if dim > 1 else 1,
        "prev_pow2": prev_p2,
        "max_rel_err": float(relerr.max()),
        "mismatch_count_total": int(mismatch.sum()),
        "mismatch_count_head_lt_prevpow2": int((mismatch & ~tail_idx).sum()),
        "mismatch_count_tail_ge_prevpow2": int((mismatch & tail_idx).sum()),
        "nonfinite_count": int((~torch.isfinite(o1)).sum()),
        "run_to_run_max_abs_diff": float((o1-o2).abs().max()),
    }
print(json.dumps({"per_dim": res, "metric": "mismatch counts vs fp64 reference split head/tail, plus nonfinite and run-to-run stability", "tol_rel": 1e-5}))
