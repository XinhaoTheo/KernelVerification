import sys, struct, json
import numpy as np, torch
sys.path.insert(0, '/root/cases/case_149')
import kernel

bits = np.arange(65536, dtype=np.uint16)
h = bits.view(np.float16)
finite = np.isfinite(h)
hf = h[finite].copy(); bf = bits[finite].copy()
N = hf.size

xs = hf.astype(np.longdouble)
ax = np.abs(xs)
e = np.exp(-ax)
sig = np.where(hf < 0, e/(np.longdouble(1)+e), np.longdouble(1)/(np.longdouble(1)+e))
r16 = np.array([struct.unpack('<e', struct.pack('<e', float(s)))[0] for s in sig], dtype=np.uint16)
rv = r16.view(np.float16).astype(np.longdouble)
expb = (r16 >> 10) & 0x1F
ulp = np.where(expb == 0, np.ldexp(np.longdouble(1), -24),
               np.ldexp(np.longdouble(1), (expb.astype(np.int32) - 1) - 10))
dist_mid = np.abs(np.abs(sig - rv) - ulp/2)  # 0 == exactly at an fp16 midpoint
rel_tie_dist = dist_mid / np.maximum(np.abs(sig), np.ldexp(np.longdouble(1), -1074))

within_fp64 = int((rel_tie_dist < np.ldexp(np.longdouble(1), -52)).sum())
within_loose = int((rel_tie_dist < np.ldexp(np.longdouble(1), -23)).sum())
min_rel = float(np.min(rel_tie_dist)); argmin = int(np.argmin(rel_tie_dist))

x = torch.from_numpy(hf).cuda()
kout = kernel.run(x).cpu().numpy().view(np.float16)
rout = kernel.reference(x).cpu().numpy().view(np.float16)
kb = kout.view(np.uint16); rb = rout.view(np.uint16)
mism = kb != rb
n_mism = int(mism.sum())

# classify mismatches: does kernel or reference match the correctly-rounded true value?
k_right = int((mism & (kb == r16)).sum())
r_right = int((mism & (rb == r16)).sum())
both_wrong = int((mism & (kb != r16) & (rb != r16)).sum())
# for mismatches, tie proximity
tie_near_mism = int((mism & (rel_tie_dist < np.ldexp(np.longdouble(1), -23))).sum())

# also compare reference's fp64 path vs longdouble truth over whole domain (reference self-accuracy)
ref_vs_true_total = int((rb != r16).sum())

examples = []
for i in np.nonzero(mism)[0][:20]:
    examples.append({"x": float(hf[i]), "kernel_bits": int(kb[i]), "ref_bits": int(rb[i]),
                     "true_bits": int(r16[i]),
                     "rel_tie_dist": float(rel_tie_dist[i])})

print(json.dumps({
    "claim": "c3", "metric": "distance of true sigmoid to exact fp16 ties (longdouble truth) and per-mismatch attribution vs true rounding",
    "n_inputs": int(N),
    "inputs_with_true_sigmoid_within_fp64_tol_of_exact_tie": within_fp64,
    "inputs_within_1_fp32_ulp_rel_of_tie": within_loose,
    "min_rel_tie_distance": min_rel, "argmin_x": float(hf[argmin]),
    "run_vs_ref_mismatches": n_mism,
    "mismatches_where_kernel_matches_true": k_right,
    "mismatches_where_ref_matches_true": r_right,
    "mismatches_both_wrong": both_wrong,
    "mismatches_near_tie_within_fp32_ulp": tie_near_mism,
    "ref_vs_true_mismatches_total_domain": ref_vs_true_total,
    "examples": examples,
}, default=str))