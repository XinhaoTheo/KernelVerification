
import numpy as np, json

SEED = 730119
PERMUTATION = [40,110,94,84,77,58,57,120,82,13,85,86,113,106,39,102,19,18,112,26,71,122,61,103,46,91,79,125,44,38,47,28,50,30,97,52,53,109,119,35,1,107,43,114,117,60,74,33,25,88,78,59,81,93,27,36,108,90,23,126,95,76,31,124,98,73,37,29,32,10,48,51,17,121,100,6,22,49,11,20,34,69,115,4,68,56,5,21,66,65,92,3,15,0,118,67,55,83,105,116,99,101,42,16,123,64,104,41,87,9,62,8,12,127,14,24,75,70,80,7,54,45,96,2,63,89,111,72]

rng = np.random.Generator(np.random.PCG64(SEED))
w = rng.integers(1,33,size=(64,32)).astype(np.float32)*np.float32(1048576)
small = rng.integers(1,4,size=(64,64)).astype(np.float32)*np.float32(0.25)
values = np.concatenate([w,-w,small],axis=1)
order = rng.permutation(128)
values = values[:,order][:,PERMUTATION].copy().astype(np.float32)

# float64 reference (values are exact in fp32; sum in float64 = real arithmetic here since all values are exact)
ref64 = values.astype(np.float64).sum(axis=1)
# cross-check with Kahan float64
kah = np.zeros(64); c = np.zeros(64)
v64 = values.astype(np.float64)
for j in range(128):
    y = v64[:,j]-c
    t = kah+y
    c = (t-kah)-y
    kah = t
assert np.allclose(ref64, kah, rtol=0, atol=1e-9)

# faithful left-to-right fp32 sequential emulation
acc = np.zeros(64, dtype=np.float32)
for j in range(128):
    acc = (acc + values[:,j]).astype(np.float32)

err = acc.astype(np.float64)-ref64
E = np.linalg.norm(err)/max(np.linalg.norm(ref64), 0.008)
per_row_rel = np.abs(err)/np.maximum(np.abs(ref64),1e-12)

print(json.dumps({
 "metric":"E = ||fp32_seq - f64_ref||_2 / max(||ref||_2, 0.008), plus per-row absolute/relative errors",
 "E": float(E), "tolerance": 0.1,
 "max_abs_err": float(np.abs(err).max()),
 "min_ref_row": float(np.abs(ref64).min()), "max_ref_row": float(np.abs(ref64).max()),
 "max_row_rel_err": float(per_row_rel.max()),
 "rows_exceeding_rel_0.1": int((per_row_rel>0.1).sum()),
 "nonfinite_emulated": bool((~np.isfinite(acc)).sum()>0),
 "kahan_crosscheck_ok": True
}))
