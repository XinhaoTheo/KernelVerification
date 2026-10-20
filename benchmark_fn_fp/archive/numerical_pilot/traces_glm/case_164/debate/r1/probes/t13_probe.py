
import torch, numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/pilot_cases/case_101/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
a, b = k.make_inputs("cuda")
out = k.run(a, b)
af = a.cpu().numpy(); bf = b.cpu().numpy()  # fp32 inputs, same values
T, D = af.shape
# Independent emulation of fp16-snapped recurrence: fp32 ops, fp16 snap each step
sim = np.empty((T,D), dtype=np.float32); h = np.zeros(D, dtype=np.float32)
for t in range(T):
    h = (af[t].astype(np.float32)*h + bf[t].astype(np.float32))
    h = h.astype(np.float16).astype(np.float32)
    sim[t] = h
o = out.cpu().numpy()
mismatch = int((o != sim).sum())
# also compare row0 (h[0]=b[0]) and step-wise recompute check
print(json.dumps({"shape": list(o.shape), "exact_mismatch_vs_fp16_sim": mismatch, "max_abs_diff_vs_sim": float(np.abs(o.astype(np.float64)-sim.astype(np.float64)).max()), "row0_matches_b0": bool(np.array_equal(o[0], bf[0].astype(np.float16).astype(np.float32))), "all_finite": bool(np.isfinite(o).all())}))
