
import torch, numpy as np, json, sys, struct
sys.path.insert(0, "/root/cases/case_149")
import kernel as K

# All finite FP16 encodings, full domain (fix earlier assertion: 65536-2048 = 63488)
codes = np.array([c for c in range(0x10000) if (c & 0x7C00) != 0x7C00], dtype=np.uint16)
x = torch.from_numpy(codes.view(np.int16).copy()).view(torch.float16).cuda().contiguous()
out = K.run(x)
ref = K.reference(x)
ob = out.cpu().numpy().view(np.uint16).astype(int)
rb = ref.cpu().numpy().view(np.uint16).astype(int)
mism = ob != rb
idx = np.nonzero(mism)[0]
details = []
for i in idx:
    xv = float(x[i].item())
    # oracle spot check: independent mpmath-free FP64 sigmoid + RNE to FP16 via numpy
    v64 = float(np.longdouble(1.0) / (np.longdouble(1.0) + np.exp(-np.longdouble(xv))))
    alt = np.array([v64], dtype=np.float16).view(np.uint16)[0]
    details.append({
        "x_bits": int(codes[i]), "x": xv,
        "got_bits": int(ob[i]), "ref_bits": int(rb[i]), "alt_bits": int(alt),
        "got_minus_ref_ulp": int(ob[i]) - int(rb[i]),
        "adjacent_ulp": abs(int(ob[i]) - int(rb[i])) == 1})
print(json.dumps({
    "claim": "c1 full-domain exhaustive sweep (all 63488 finite FP16 encodings)",
    "metric": "exact FP16 bit-pattern equality vs reference() oracle, plus independent FP64/numpy-RNE spot check of mismatching values",
    "n_total": int(len(codes)),
    "mismatch_count": int(mism.sum()),
    "n_negative_mismatches": int((mism & (codes[idx.astype(bool)] if False else np.array([(int(c) & 0x8000) != 0 for c in codes]))).sum()) if len(idx)>0 else 0,
    "details": details,
    "verdict_supports_bug": int(mism.sum()) > 0}))
