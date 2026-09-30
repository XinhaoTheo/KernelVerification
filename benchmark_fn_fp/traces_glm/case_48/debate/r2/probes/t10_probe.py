import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_m")
import kernel as K

coeffs, pts = K.make_inputs(device="cuda")
out = K.run(coeffs, pts)
torch.cuda.synchronize()

# float64 reference over stored float32 values, Horner in float64
c64 = coeffs.cpu().numpy().astype(np.float64)
p64 = pts.cpu().numpy().astype(np.float64)
ref = np.zeros(8, dtype=np.float64)
for i in range(8):
    acc = c64[i, 48]
    for k in range(47, -1, -1):
        acc = acc * p64[i] + c64[i, k]
    ref[i] = acc

o = out.cpu().numpy().astype(np.float64)
diff = o - ref
rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(ref), 0.001*np.sqrt(8))
finite = bool(np.all(np.isfinite(o)))

# also emulate pure float32 Horner (separate rounding) on CPU for cross-check
c32 = coeffs.cpu().numpy(); p32 = pts.cpu().numpy()
emu = np.zeros(8, dtype=np.float32)
max_inter = 0.0
for i in range(8):
    acc = np.float32(c32[i,48])
    mi = abs(acc)
    for k in range(47,-1,-1):
        prod = np.float32(np.float32(acc) * np.float32(p32[i]))
        acc = np.float32(prod + c32[i,k])
        mi = max(mi, abs(acc))
    emu[i] = acc
    max_inter = max(max_inter, mi)

print(json.dumps({
    "kernel_output": o.tolist(),
    "reference": ref.tolist(),
    "abs_err": diff.tolist(),
    "rel_l2": float(rel_l2),
    "tolerance": 0.0002,
    "passes_contract": bool(rel_l2 <= 0.0002 and finite),
    "finite": finite,
    "ref_norm": float(np.linalg.norm(ref)),
    "floor": 0.001*8**0.5,
    "c0_magnitudes": np.abs(c32[:,0]).tolist(),
    "max_intermediate_abs": max_inter,
    "emu_rel_l2": float(np.linalg.norm(emu.astype(np.float64)-ref)/max(np.linalg.norm(ref),0.001*np.sqrt(8)))
}))
