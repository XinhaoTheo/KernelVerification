import json, numpy as np, torch, sys
sys.path.insert(0, "/root/numerical_cases/case_n")
import kernel as K

coefficients, points = K.make_inputs("cuda")
out = K.run(coefficients, points)
torch.cuda.synchronize()

# float64 reference from stored float32 values
c64 = coefficients.detach().cpu().numpy().astype(np.float64)
p64 = points.detach().cpu().numpy().astype(np.float64)
ref = np.array([np.sum(c64[i] * p64[i] ** np.arange(49)) for i in range(8)])

outn = out.detach().cpu().numpy().astype(np.float64)
err_norm = np.linalg.norm(outn - ref)
ref_norm = np.linalg.norm(ref)
floor = 0.001 * np.sqrt(8)
denom = max(ref_norm, floor)
rel = err_norm / denom
result = {
    "per_row_reference": ref.tolist(),
    "per_row_kernel": outn.tolist(),
    "per_row_abs_err": (outn - ref).tolist(),
    "err_L2": float(err_norm),
    "ref_L2": float(ref_norm),
    "floor_0p001sqrt8": float(floor),
    "denominator": float(denom),
    "relative_L2": float(rel),
    "tolerance": 0.0002,
    "exceeds_tolerance": bool(rel > 0.0002),
    "floor_active": bool(ref_norm < floor),
    "all_finite": bool(np.all(np.isfinite(outn))),
    "shape": list(outn.shape),
}
print(json.dumps(result))
