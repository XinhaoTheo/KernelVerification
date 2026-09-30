import sys, json
import numpy as np, torch
sys.path.insert(0, "/root/pilot_cases/case_85")
import kernel

x, w = kernel.make_inputs("cuda")
wf64 = w.double().cpu().numpy()          # exact float32 weights as reals
w32 = w.cpu().numpy()                    # float32

# fp64 quantizer (as used in make_inputs' residual construction)
s64 = np.max(np.abs(wf64), axis=1, keepdims=True) / 7.0
qi64 = np.clip(np.floor(wf64 / s64 + 0.5), -7, 7)

# simulate kernel's fp32 quantizer exactly
s32 = (np.float32(np.max(np.abs(w32), axis=1, keepdims=True)) / np.float32(7.0))
t32 = w32 / s32 + np.float32(0.5)
qi32 = np.clip(np.floor(t32), np.float32(-7), np.float32(7)).astype(np.float32)

flips = int((qi64.astype(np.float32) != qi32).sum())
s_mismatch = int((s64.astype(np.float32) != s32).sum())

# does flipping change the error vector vs fp64 sim? compare reconstructed weights
wq32 = qi32 * s32
wq64 = qi64 * s64
xf = x.double().cpu().numpy()
ref = wf64 @ xf
E32 = np.linalg.norm(wq32 @ xf - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(32))
E64 = np.linalg.norm(wq64 @ xf - ref) / max(np.linalg.norm(ref), 0.001*np.sqrt(32))

# distance of any element to a .5 boundary in fp32
dist_to_boundary = np.abs((t32 - np.floor(t32)) - np.float32(0.5))
min_dist = float(dist_to_boundary.min())

print(json.dumps({
    "metric": "fp32 vs fp64 quantizer level flips on fixed W (4096 elems)",
    "num_elements": int(wf64.size),
    "qi_level_flips": flips,
    "scale_mismatches_rows": s_mismatch,
    "min_distance_to_.5_boundary_fp32": min_dist,
    "E_fp32_quantized_matvec": float(E32),
    "E_fp64_quantized_matvec": float(E64),
    "E_budget": 0.12,
}))