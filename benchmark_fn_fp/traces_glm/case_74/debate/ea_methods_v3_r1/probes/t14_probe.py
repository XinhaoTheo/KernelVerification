import json, math, numpy as np, sys
sys.path.insert(0, "/root/evidence_cases/case_e13")
import kernel

x, = kernel.make_inputs_numpy()
# trace per-row: which smalls end up in final correction, which dropped
traces = []
outputs = []
for r in range(4):
    t = np.float32(0.0); c = np.float32(0.0)
    snap = []
    for j, v in enumerate(x[r]):
        v = np.float32(v)
        u = np.float32(t+v)
        if abs(t) >= abs(v):
            lost = np.float32(np.float32(t-u)+v)
        else:
            lost = np.float32(np.float32(v-u)+t)
        c = np.float32(c+lost)
        t = u
        snap.append({"j": j, "v": float(v), "lost": float(lost), "corr_after": float(c)})
    out = np.float32(t+c)
    smalls = [float(v) for v in x[r] if abs(float(v)) <= 2.0]
    kept = sum(s["lost"] for s in snap if abs(s["v"]) <= 2.0)  # algebraic, not float
    traces.append({"row": r, "final_total": float(t), "final_correction": float(c), "output": float(out), "smalls_sum_exact": math.fsum(smalls), "column_order": [float(v) for v in x[r]]})
    outputs.append(float(out))
vals = x.astype(np.float64)
targets = [math.fsum(vals[r].tolist()) for r in range(4)]
err = float(np.linalg.norm(np.array(outputs)-np.array(targets))/max(np.linalg.norm(np.array(targets)),1e-12))
print(json.dumps({"traces": traces, "exact_targets": targets, "simulated_outputs": outputs, "relative_error": err, "tolerance": 1e-5, "drops_smalls": bool(err > 1e-5)}))