
import torch, json, importlib.util, math
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_02/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

def ref64(ns, da):
    ns = ns.double().cpu(); da = da.double().cpu()
    st = torch.zeros(ns.shape[1], dtype=torch.float64)
    for c in range(ns.shape[0]):
        st = torch.exp(da[c]) * st + ns[c]
    return st

def subdivide(ns, da, K):
    nch, dim = ns.shape
    ns2 = torch.zeros(nch*K, dim)
    da2 = torch.zeros(nch*K)
    for c in range(nch):
        # split log-decay evenly; local state on LAST sub-chunk
        da2[c*K:(c+1)*K] = da[c] / K
        ns2[(c+1)*K - 1] = ns[c]
    return ns2, da2

torch.manual_seed(1)
out = {}

# A) pure exp-compounding test: state=1 then K multiplications of exp(d/K)
expcases = []
for d in [-0.5, -5.0, -20.0, 1.0]:
    for K in [1, 10, 100, 1000]:
        nch = K+1
        ns = torch.zeros(nch, 4); ns[0] = 1.0
        da = torch.full((nch,), d/K); da[0] = 0.0
        got = k.state_passing(ns.cuda(), da.cuda()).cpu().double()
        exact = math.exp(d)
        expcases.append(dict(d=d, K=K, got=float(got[0]), exact=exact,
                             rel=abs(float(got[0])-exact)/abs(exact),
                             rel_per_step=abs(float(got[0])-exact)/abs(exact)/K))
out["exp_compounding"] = expcases

# B) full subdivision invariance on random problems
subcases = []
for nch, dim, K, scale in [(8,64,1,1.0),(8,64,4,1.0),(8,64,16,1.0),(8,64,64,1.0),
                           (16,32,8,3.0),(4,128,256,1.0)]:
    ns = torch.randn(nch, dim)
    da = -torch.rand(nch)*scale
    base = k.state_passing(ns.cuda(), da.cuda()).cpu().double()
    ns2, da2 = subdivide(ns, da, K)
    sub = k.state_passing(ns2.cuda(), da2.cuda()).cpu().double()
    r = ref64(ns, da)
    r2 = ref64(ns2, da2)
    subcases.append(dict(nchunks=nch, dim=dim, K=K, decay_scale=scale,
        ref_consistency_max_abs=float((r-r2).abs().max()),
        ref_absmax=float(r.abs().max()),
        base_vs_ref_maxabs=float((base-r).abs().max()),
        sub_vs_ref_maxabs=float((sub-r).abs().max()),
        base_vs_sub_maxabs=float((base-sub).abs().max()),
        sub_vs_ref_rel_to_scale=float((sub-r).abs().max()/max(1e-30,float(r.abs().max()))),
    ))
out["subdivision"] = subcases
out["metric"] = "compounded exp relative error; subdivided-vs-original kernel outputs both vs shared float64 reference"
print(json.dumps(out, indent=1))
