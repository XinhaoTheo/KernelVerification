import sys, json
sys.path.insert(0, "/root/numerical_cases/case_o")
import numpy as np, torch
import kernel as K

u, b = K.make_inputs()
u32 = u.cpu().numpy(); b32 = b.cpu().numpy()

# sequential float32 accumulation exactly as kernel does (separately rounded products/sums)
num = np.float32(0.0); den = np.float32(0.0)
for j in range(32):
    num = np.float32(num + np.float32(u32[j]*b32[j]))
    den = np.float32(den + np.float32(u32[j]*u32[j]))
alpha_f32 = np.float32(num/den)

u64 = u32.astype(np.float64); b64 = b32.astype(np.float64)
alpha_f64 = (u64*b64).sum()/(u64*u64).sum()

rel_err = abs(float(alpha_f32)-alpha_f64)/abs(alpha_f64)
print(json.dumps({
  "metric": "relative error of sequentially accumulated float32 alpha vs float64 alpha",
  "alpha_f32_sequential": float(alpha_f32),
  "alpha_f64": alpha_f64,
  "alpha_rel_err": float(rel_err),
  "c2_threshold": 1e-5,
  "numerator_f32": float(num),
  "denominator_f32": float(den),
  "numerator_f64": float((u64*b64).sum()),
  "denominator_f64": float((u64*u64).sum()),
}))