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
rout = kernel.reference(x).cpu().numpy().view(np.float16)
kb = kout.view(np.uint16); rb = rout.view(np.uint16)
mism = kb != rb
n_mism = int(mism.sum())

# true real sigmoid in longdouble, then direct RNE pack to fp16
xs = hf.astype(np.longdouble)
ax = np.abs(xs)
e = np.exp(-ax)  # longdouble exp
sig = np.where(hf < 0, e/(np.longdouble(1)+e), np.longdouble(1)/(np.longdouble(1)+e))
def pack_e(v):
    return struct.unpack('<H', struct.pack('<e', float(v)))[0]
true_bits = np.array([pack_e(s) for s in sig], dtype=np.uint16)

examples = []
idxs = np.nonzero(mism)[0][:20]
for i in idxs:
    examples.append({
        "x_fp16": float(hf[i]), "x_bits": int(bf[i]),
        "kernel_bits": int(kb[i]), "ref_bits": int(rb[i]),
        "true_bits": int(true_bits[i]),
        "kernel_matches_true": bool(kb[i] == true_bits[i]),
        "ref_matches_true": bool(rb[i] == true_bits[i]),
    })
kernel_wrong = int((kb != true_bits).sum())
ref_wrong = int((rb != true_bits).sum())

print(json.dumps({
    "claim": "c1", "metric": "exact fp16 bit equality run vs reference over all finite fp16 inputs",
    "n_inputs": int(N), "mismatches_run_vs_ref": n_mism,
    "kernel_vs_true_mismatches": kernel_wrong,
    "ref_vs_true_mismatches": ref_wrong,
    "examples": examples,
}, default=str))