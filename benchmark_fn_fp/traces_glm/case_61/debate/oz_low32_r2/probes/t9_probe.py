import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_z")
import kernel

q, a, v = kernel.make_inputs_numpy()
qt, at, vt = kernel.make_inputs()
out = kernel.run(qt, at, vt).detach().cpu().numpy()

# emulate kernel's float32 expanded identity with per-step rounding (numpy float32 ops round per op)
rowwise = []
for i in range(16):
    anchor_norm = np.float32(0.0); query_norm = np.float32(0.0); dot = np.float32(0.0)
    for k in range(32):
        ak = np.float32(a[i,k]); qk = np.float32(q[k])
        anchor_norm = np.float32(anchor_norm + np.float32(ak*ak))
        query_norm = np.float32(query_norm + np.float32(qk*qk))
        dot = np.float32(dot + np.float32(ak*qk))
    norm_sum = np.float32(anchor_norm + query_norm)
    twice = np.float32(np.float32(2.0)*dot)
    pre = np.float32(norm_sum - twice)
    rowwise.append(float(pre))
pre = np.array(rowwise)
neg = pre < 0
w_kernel = np.exp(-16.0*np.maximum(pre,0.0)).astype(np.float64)

d64 = ((a.astype(np.float64) - q.astype(np.float64)[None,:])**2).sum(axis=1)
w64 = np.exp(-16.0*d64)
ref = (w64*v.astype(np.float64)).sum()/w64.sum()

print(json.dumps({
  "pre_clamp_distances": [round(x,6) for x in rowwise],
  "num_negative_rows": int(neg.sum()),
  "min_pre_clamp": float(pre.min()),
  "emulated_clamp_output": float((w_kernel*v.astype(np.float64)).sum()/w_kernel.sum()),
  "reference": float(ref),
  "kernel_output": float(out[0]),
  "relative_error_kernel": abs(float(out[0])-float(ref))/max(abs(ref),0.05)
}))