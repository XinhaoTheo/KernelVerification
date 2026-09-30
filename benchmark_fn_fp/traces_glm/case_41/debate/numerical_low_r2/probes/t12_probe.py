import sys, json, torch
sys.path.insert(0, "/root/numerical_cases/case_f")
import kernel as K

matrix, initial, drive = K.make_inputs()
out = K.run(matrix, initial, drive)

m64 = matrix.double(); i64 = initial.double(); d64 = drive.double()
ref = i64.clone()
for t in range(64):
    ref = m64 @ ref + d64[t]

den = max(ref.norm().item(), 0.001*16**0.5)
err = (out.double() - ref).norm().item()
rel = err / den

# diagnostic: fp16 cast emulation in fp64 arithmetic
s = i64.clone()
for t in range(64):
    s = (m64 @ s + d64[t]).to(torch.float16).double()
sim_err = (s - ref).norm().item()

print(json.dumps({
    "kernel_rel_l2": rel,
    "abs_err": err,
    "denominator": den,
    "ref_norm": ref.norm().item(),
    "tolerance": 0.002,
    "fp16_sim_rel_l2": sim_err / den,
    "finite": bool(torch.isfinite(out).all().item()),
    "shape": list(out.shape),
    "exceeds_tolerance": rel > 0.002,
}))