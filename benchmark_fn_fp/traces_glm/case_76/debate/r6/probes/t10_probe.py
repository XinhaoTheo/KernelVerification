import sys, struct, json
import numpy as np, torch
sys.path.insert(0, '/root/cases/case_149')
import kernel

bits = np.arange(65536, dtype=np.uint16)
h = bits.view(np.float16)
finite = np.isfinite(h)
hf = h[finite].copy(); bf = bits[finite].copy()
N = hf.size

x = torch.from_numpy(hf).cuda()
kout = kernel.run(x).cpu().numpy().view(np.float16)
kb = kout.view(np.uint16)

# oracle FP32 pipeline with correctly-rounded exp: e32 = RNE(exp(-|x|))
xs = hf.astype(np.longdouble)
ax = np.abs(xs)
e_ld = np.exp(-ax)
e32 = e_ld.astype(np.float32)          # single rounding to fp32 (best exp can be)
one = np.float32(1.0)
d32 = (one + e32).astype(np.float32)   # fp32 add, RNE
num = np.where(hf < 0, e32, one).astype(np.float32)
v32 = (num / d32).astype(np.float32)   # fp32 divide, RNE
def pack_e(v):
    return struct.unpack('<H', struct.pack('<e', float(v)))[0]
oracle_bits = np.array([pack_e(v) for v in v32], dtype=np.uint16)

# midpoint proximity of TRUE sigmoid (longdouble): distance to nearest fp16 midpoint
sig = np.where(hf < 0, e_ld/(np.longdouble(1)+e_ld), np.longdouble(1)/(np.longdouble(1)+e_ld))
r16 = np.array([struct.unpack('<e', struct.pack('<e', float(s)))[0] for s in sig], dtype=np.uint16)
rv = r16.view(np.float16).astype(np.longdouble)
# ulp of the rounded fp16 value (next representable)
ulp = np.where((r16.view(np.uint16) & 0x7FFF) < 0x7BFF,
               np.ldexp(np.longdouble(1), (r16.view(np.int16) - np.int16(15)) - 1),
               np.ldexp(np.longdouble(1), -24))  # fallback, rarely used
# for subnormal outputs ulp = 2^-24; fix via exponent bits
expb = (r16.view(np.uint16) >> 10) & 0x1F
ulp = np.where(expb == 0, np.ldexp(np.longdouble(1), -24),
               np.ldexp(np.longdouble(1), (expb.astype(np.int32) - 1) - 10))
dist_to_nearest_mid = np.abs(sig - rv) - np.abs(ulp/2)  # signed distance from midpoint band edge... use |sig-rv| vs ulp/2
dist_mid = np.abs(sig - rv) - ulp/2  # <0 means inside rounding interval; distance to the *boundary* midpoint
abs_dist_mid = np.abs(dist_mid)
# relative distance to nearest midpoint
rel_dist = abs_dist_mid / np.maximum(np.abs(sig), np.ldexp(np.longdouble(1), -1074))
within_fp32ulp = int((rel_dist < np.ldexp(np.longdouble(1), -23)).sum())
min_rel = float(np.min(rel_dist))
argmin = int(np.argmin(rel_dist))

# comparisons
k_vs_oracle = int((kb != oracle_bits).sum())
oracle_vs_true = int((oracle_bits != r16).sum())
# kernel mismatches vs oracle, and whether oracle matches reference there
rout = kernel.reference(x).cpu().numpy().view(np.float16)
rb = rout.view(np.uint16)
k_ne_or = kb != oracle_bits
exp_induced = int((k_ne_or & (oracle_bits == rb)).sum())

examples = []
for i in np.nonzero(k_ne_or)[0][:20]:
    examples.append({"x": float(hf[i]), "kernel_bits": int(kb[i]),
                     "oracle_bits": int(oracle_bits[i]), "ref_bits": int(rb[i]),
                     "true_bits": int(r16[i]),
                     "rel_dist_to_midpoint": float(rel_dist[i])})

print(json.dumps({
    "claim": "c2", "metric": "kernel bits vs oracle fp32 pipeline with correctly-rounded exp; plus true-sigmoid proximity to fp16 midpoints",
    "n_inputs": int(N),
    "kernel_vs_oracle_mismatches": k_vs_oracle,
    "oracle_vs_true_mismatches": oracle_vs_true,
    "exp_induced_mismatches (kernel!=oracle, oracle==ref)": exp_induced,
    "inputs_with_true_sigmoid_within_1_fp32_ulp_rel_of_midpoint": within_fp32ulp,
    "min_rel_dist_to_midpoint": min_rel,
    "argmin_x": float(hf[argmin]), "argmin_rel_dist": float(rel_dist[argmin]),
    "examples": examples,
}, default=str))