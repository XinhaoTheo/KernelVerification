import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_m")
import kernel as K

coeffs, pts = K.make_inputs(device="cuda")
out = K.run(coeffs, pts)
torch.cuda.synchronize()

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

c32 = coeffs.cpu().numpy(); p32 = pts.cpu().numpy()
emu = np.zeros(8, dtype=np.float32)
max_inter = 0.0
for i in range(8):
    acc = np.float32(c32[i,48])
    mi = abs(float(acc))
    for k in range(47,-1,-1):
        prod = np.float32(np.float32(acc) * np.float32(p32[i]))
        acc = np.float32(prod + c32[i,k])
        mi = max(mi, abs(float(acc)))
    emu[i] = acc
    max_inter = max(max_inter, mi)

print(json.dumps({
    "kernel_output": [float(x) for x in o],
    "reference": [float(x) for x in ref],
    "abs_err": [float(x) for x in diff],
    "rel_l2": float(rel_l2),
    "tolerance": 0.0002,
    "passes_contract": bool(rel_l2 <= 0.0002 and finite),
    "finite": finite,
    "ref_norm": float(np.linalg.norm(ref)),
    "floor": 0.001*8**0.5,
    "c0_magnitudes": [float(x) for x in np.abs(c32[:,0])],
    "max_intermediate_abs": float(max_inter),
    "emu_rel_l2": float(np.linalg.norm(emu.astype(np.float64)-ref)/max(np.linalg.norm(ref),0.001*np.sqrt(8)))
}))