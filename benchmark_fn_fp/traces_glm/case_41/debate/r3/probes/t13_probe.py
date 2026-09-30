import sys, json, torch
sys.path.insert(0, "/root/numerical_cases/case_f")
import kernel as K

matrix, initial, drive = K.make_inputs()
m64 = matrix.double(); i64 = initial.double(); d64 = drive.double()
ref = i64.clone()
norms = []
for t in range(64):
    ref = m64 @ ref + d64[t]
    norms.append(ref.norm().item())
final_norm = ref.norm().item()
floor = 0.001 * 16 ** 0.5
den = max(final_norm, floor)

# smallest fp16 error magnitude available: quantization step at scale of ref entries
out = K.run(matrix, initial, drive)
abs_err = (out.double() - ref).norm().item()
rel = abs_err / den
# hypothetical rel error if denominator were the floor
rel_if_floor = abs_err / floor

print(json.dumps({
    "ref_final_norm": final_norm,
    "floor": floor,
    "denominator_is_floor": final_norm <= floor,
    "norm_ratio_to_floor": final_norm / floor,
    "min_state_norm_over_steps": min(norms),
    "kernel_abs_err": abs_err,
    "kernel_rel_l2_actual_den": rel,
    "kernel_rel_l2_if_floor_den": rel_if_floor,
    "max_component_abs": ref.abs().max().item(),
    "tolerance": 0.002,
}))