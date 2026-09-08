
import torch, json, importlib.util, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_02/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref(ns, da):
    ns = ns.double().cpu(); da = da.double().cpu()
    st = torch.zeros(ns.shape[1], dtype=torch.float64)
    for c in range(ns.shape[0]):
        st = torch.exp(da[c]) * st + ns[c]
    return st

res = []
torch.manual_seed(0)
cases = [
    ("small_neg", 4, 8, lambda n: -torch.rand(n)),
    ("dim1", 16, 1, lambda n: -torch.rand(n)),
    ("dim_nonpow2", 8, 100, lambda n: -torch.rand(n)*2),
    ("dim3", 5, 3, lambda n: -torch.rand(n)),
    ("many_chunks", 1024, 64, lambda n: -torch.rand(n)*0.01),
    ("mixed_sign", 32, 64, lambda n: (torch.rand(n)-0.5)*0.5),
    ("strong_decay", 64, 128, lambda n: -torch.rand(n)*5),
    ("zero_decay", 64, 64, lambda n: torch.zeros(n)),
    ("big_dim", 8, 1024, lambda n: -torch.rand(n)),
]
for name, nch, dim, daf in cases:
    ns = torch.randn(nch, dim)
    da = daf(nch)
    out = k.state_passing(ns.cuda(), da.cuda()).cpu().double()
    r = ref(ns, da)
    err = (out - r).abs()
    denom = r.abs().clamp_min(1e-30)
    res.append(dict(case=name, nchunks=nch, dim=dim,
                    max_abs=float(err.max()), max_rel=float((err/denom).max()),
                    ref_absmax=float(r.abs().max()), out_dtype=str(out.dtype),
                    out_shape=list(out.shape)))
print(json.dumps({"metric":"max abs/rel err vs float64 recurrence","cases":res}, indent=1))
