
import json, sys
sys.path.insert(0, "/root/cases/case_141")
import torch
from kernel import run, reference, _SHIFT, _MULTIPLIER

# Part 1: exhaustive exact check of the formula floor((v*M)>>S) == v//7 over all 2^32 values
M, S, D = _MULTIPLIER, _SHIFT, 7
print("shift", S, "multiplier", M)
worst = None
# v*M has ~66 bits; use Python ints on representative + scanned ranges
# scan boundaries: v*M>>S first potentially differs from v//7 when 7*(v*M>>S) > v
# error term: q = floor(v*M/2^S); v*M = v*(2^S+e-... ) let e = D*M - 2^S
e = D*M - (1 << S)
print("e =", e)
# v*M - v*2^S/7 ... check directly in chunks + full fast loop is 2^32 heavy in Python (~minutes). Use numpy vectorized instead.
import numpy as np
bad = 0
first_bad = None
# chunked numpy with object-free math: v*M up to ~2^66 overflows uint64 (M~2^31.2, v<2^32 -> product <2^64? M=2457124170 < 2^32, v<2^32 -> product <2^64, fits uint64!). Check:
prod_max = ( (1<<32) - 1 ) * M
print("max product bits:", prod_max.bit_length())
assert prod_max < (1<<64)
chunk = 1 << 24
vbase = 0
while vbase < (1 << 32):
    v = np.arange(vbase, min(vbase+chunk, 1<<32), dtype=np.uint64)
    q = (v * np.uint64(M)) >> np.uint64(S)
    ref = v // np.uint64(D)
    d = q != ref
    nb = int(d.sum())
    if nb:
        bad += nb
        if first_bad is None:
            first_bad = int(v[d][0])
    vbase += chunk
print("pure-formula mismatches over full uint32 domain:", bad, "first_bad:", first_bad)

# Part 2: real kernel on worst-case / boundary values
vals = [0, 1, 6, 7, 8, 13, 14, 49, 2**31, 2**31+1,
        4294967295, 4294967294, 4294967293, 4294967289,
        4294967295-7, 4294967295-14, 2**32-1-100]
if first_bad is not None:
    vals += [first_bad, first_bad-1, first_bad+1, first_bad+7]
# also a dense random sample incl high values
vals += list(np.random.default_rng(1).integers(2**31, 2**32, 1000).tolist())
u = torch.tensor(sorted(set(vals)), dtype=torch.uint32, device="cuda")
out = run(u)
ref = reference(u.cpu()).cuda()
mism = (out != ref).any(dim=1)
n_mism = int(mism.sum())
print("kernel mismatches on stress set:", n_mism)
if n_mism:
    idx = mism.nonzero().flatten()[:10]
    for i in idx.tolist():
        v = int(u[i])
        print("v=", v, "kernel=", out[i].tolist(), "ref=", ref[i].tolist(),
              "7*row+col=", 7*out[i,0].item()+out[i,1].item())

# also full random sweep with make_inputs
from kernel import make_inputs
mis_rand = 0
for seed in range(5):
    x, = make_inputs(seed=seed)
    o = run(x); r = reference(x.cpu()).cuda()
    mis_rand += int((o != r).any(dim=1).sum())
print("random-sample mismatches:", mis_rand)
print(json.dumps({"formula_full_domain_bad": bad, "first_bad": first_bad,
                  "kernel_stress_mismatch": n_mism, "kernel_random_mismatch": mis_rand}))
