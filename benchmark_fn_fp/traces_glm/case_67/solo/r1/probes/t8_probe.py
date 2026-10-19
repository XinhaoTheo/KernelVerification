import torch, json, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("kernel_mod", "/root/cases/case_140/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
dev = "cuda"

vals = set([0,1,6,7,8,13,14,4294967289,4294967290,4294967295])
for q in [0,1,2,306783378,612712399,613566756,(2**32-1)//7]:
    for v in [q*7-1,q*7,q*7+1,q*7+6,q*7+7]:
        if 0 <= v <= 4294967295: vals.add(v)
g = torch.Generator().manual_seed(0)
vals.update(torch.randint(0,2**32,(100000,),dtype=torch.int64,generator=g).tolist())
vals.update(range(0,5000)); vals.update(range(2**32-2000,2**32))
arr = torch.tensor(sorted(vals), dtype=torch.int64)

nbad = 0; inv_ok = True; first_bad = None
CH = 131072
for s in range(0, arr.numel(), CH):
    chunk = arr[s:s+CH].to(torch.uint32).to(dev)
    out = mod.run(chunk)
    v64 = chunk.to(torch.int64)
    ref = torch.stack((torch.div(v64,7,rounding_mode="floor"), v64 % 7),dim=1).to(torch.uint32)
    if not torch.equal(out, ref):
        bad = (out != ref).any(dim=1).nonzero()
        nbad += bad.numel()
        if first_bad is None:
            i = int(bad[0,0]); first_bad = {"value": int(chunk[i].item()), "kernel": out[i].cpu().tolist(), "ref": ref[i].cpu().tolist()}
    inv_ok = inv_ok and bool(((out[:,0].to(torch.int64)*7 + out[:,1].to(torch.int64)) == v64).all().item())

# Exhaustive verification of the kernel's exact formula ((v+1)*M)>>34 over ALL 2^32 uint32 values
M = np.uint64((1 << 34) // 7)
exhaustive_ok = True; ex_bad = None
CH2 = 1 << 24  # 16M per chunk
for s in range(0, 1 << 32, CH2):
    v = np.arange(s, s + CH2, dtype=np.uint64)
    q = ((v + np.uint64(1)) * M) >> np.uint64(34)
    rq = v // np.uint64(7); rr = v % np.uint64(7)
    ok = (q == rq) & ((v - q * np.uint64(7)) == rr)
    if not ok.all():
        exhaustive_ok = False
        i = int(np.argmax(~ok)); ex_bad = {"value": int(v[i]), "q": int(q[i]), "ref_q": int(rq[i])}
        break

print(json.dumps({"gpu_inputs": arr.numel(), "gpu_mismatches": nbad,
    "invariant_7row_plus_col": inv_ok, "first_bad_gpu": first_bad,
    "exhaustive_all_uint32_formula_exact": exhaustive_ok, "exhaustive_bad": ex_bad}))
