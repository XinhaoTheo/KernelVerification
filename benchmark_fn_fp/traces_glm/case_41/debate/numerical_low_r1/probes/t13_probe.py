import json, sys, torch, numpy as np
sys.path.insert(0, "/root/numerical_cases/case_f")
import kernel as K

matrix, initial, drive = K.make_inputs("cuda")

m64 = matrix.double().cpu().numpy()
s64 = initial.double().cpu().numpy().astype(np.float64)
d64 = drive.double().cpu().numpy().astype(np.float64)
ref = s64.copy()
for t in range(64):
    ref = m64 @ ref + d64[t]

out = K.run(matrix, initial, drive).cpu().numpy()
denom = max(np.linalg.norm(ref), 0.001*np.sqrt(16))
rel = np.linalg.norm(out.astype(np.float64) - ref)/denom

# error decomposition: per-step fp16 rounding of intermediate states (source line 16)
sim = s64.copy()
for t in range(64):
    nxt = m64 @ sim + d64[t]
    sim = nxt.astype(np.float16).astype(np.float64)
rel_sim = np.linalg.norm(sim - ref)/denom

print(json.dumps({
    "metric": "relative L2 vs float64 reference on fixed seeded workload",
    "reason": "contract requires rel L2 <= 0.002 for run(*make_inputs())",
    "rel_l2_kernel": float(rel),
    "rel_l2_fp16_per_step_sim": float(rel_sim),
    "tolerance": 0.002,
    "ref_norm": float(np.linalg.norm(ref)),
    "out_finite": bool(np.all(np.isfinite(out))),
    "out_shape": list(out.shape),
    "out_dtype": str(out.dtype)
}))
