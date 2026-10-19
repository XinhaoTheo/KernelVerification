import torch, importlib.util, numpy as np, math, struct, json
import mpmath as mp
mp.mp.dps = 60
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_148/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

# all legal finite FP16 encodings
bits = np.arange(65536, dtype=np.uint16)
legal = (bits.astype(np.uint16) & np.uint16(0x7C00)) != np.uint16(0x7C00)
lb = bits[legal]
xf = lb.view(np.float16)
N = len(xf)
x64 = xf.astype(np.float64)
# vectorized FP64 sigmoid
s64 = np.where(x64 >= 0, 1.0/(1.0+np.exp(-np.abs(x64))), np.exp(-np.abs(x64))/(1.0+np.exp(-np.abs(x64))))
# FP64 -> FP16 RNE reference (numpy uses RNE)
ref16 = s64.astype(np.float16)
refbits = ref16.view(np.uint16).astype(np.int64)

# identify near-tie cases for mpmath adjudication:
# distance from s64 to nearest fp16 grid relative to ulp
half = ref16.astype(np.float64)
# ulp of result (including subnormal handling)
u = np.ldexp(np.float64(2**-10), (np.frexp(ref16)[1].astype(np.int64) - 1))  # approx
# better: compute spacing around each ref value using nextafter
up = np.nextafter(ref16, np.float16(np.inf)).astype(np.float64)
dn = np.nextafter(ref16, np.float16(-np.inf)).astype(np.float64)
mid_up = 0.5*(half + up); mid_dn = 0.5*(half + dn)
dist = np.minimum(np.abs(s64-mid_up), np.abs(s64-mid_dn))
# cases where fp64 value is within 2^-38 of a tie boundary -> could round differently from exact
cand = dist < 1e-10
cand_idx = np.nonzero(cand)[0]
n_cand = len(cand_idx)

# also verify FP64 computation itself is faithful: recompute with exp in extended precision for candidates
def exact_ref_bits(v):
    t = mp.mpf(v)
    e = mp.exp(mp.fabs(t))
    s = 1/(1+e) if t >= 0 else e/(1+e)
    # round to fp16 RNE
    if s == 0: return 0x0000
    sign = 0
    if s >= mp.mpf(2)**-14:
        ex = -14
        while mp.mpf(2)**(ex+1) <= s: ex += 1
        frac = s / mp.mpf(2)**ex - 1
        step = mp.mpf(2)**-10
    else:
        ex = -14
        frac = s / mp.mpf(2)**-14
        step = mp.mpf(2)**-24
    q = frac/step
    qi = mp.floor(q)
    r = q - qi
    if r > mp.mpf('0.5'): qi = qi + 1
    elif r == mp.mpf('0.5'):
        if qi % 2 == 1: qi = qi + 1
    if ex == -14:
        return int(qi) & 0x3FF if qi < mp.mpf(2)**10 else 0x400 | (int(qi) & 0x3FF)
    m = int(qi)
    if m >= 2048: m = 1024; ex += 1
    return ((ex+15) << 10) | (m & 0x3FF)

# run kernel on all legal values (N=63488 <= 131072)
xc = torch.from_numpy(xf.copy()).cuda()
out = k.run(xc)
ob = out.view(torch.int16).to(torch.int64).cpu().numpy() & 0xFFFF
rb = refbits

bad = np.nonzero(ob != rb)[0]
n_bad = len(bad)
# adjudicate every mismatch (and near-tie candidates) with exact mpmath reference
def adjud(i):
    v = float(x64[i])
    eb = exact_ref_bits(v)
    return {"x": v, "in_bits": int(lb[i]), "kernel_bits": int(ob[i]), "fp64ref_bits": int(rb[i]), "exact_bits": eb,
            "kernel_matches_exact": int(ob[i]) == eb, "kernel_matches_fp64ref": int(ob[i]) == int(rb[i])}
examples = [adjud(int(i)) for i in bad[:20]]
cand_adj = [adjud(int(i)) for i in cand_idx[:50]]
res = {"n_legal": int(N), "mismatches_vs_fp64ref": int(n_bad),
       "n_near_tie_candidates": int(n_cand),
       "near_tie_adjudicated": cand_adj,
       "examples": examples,
       "kernel_exact_failures": int(sum(1 for i in bad if int(ob[i]) != exact_ref_bits(float(x64[i]))))}
print(json.dumps(res))
