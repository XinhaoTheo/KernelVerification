import sys, json, torch, numpy as np
sys.path.insert(0, "/root/pilot_cases/case_103")
import kernel as K

q, k, v = K.make_inputs("cuda")

# reference probabilities in float64 (matching make_inputs construction)
q64, k64 = q.double(), k.double()
z = k64 @ q64 / np.sqrt(32.0)
z = z - z.max()
p64 = torch.exp(z)
p64 = p64 / p64.sum()

# fp16 quantized probabilities (what the kernel computes before p@V)
p16 = p64.to(torch.float16).to(torch.float64)
perr = (p16 - p64).abs()

# y error attributable solely to fp16 snapping of p
v64 = v.double()
ref = p64 @ v64
y_from_p16 = p16 @ v64
err_norm = (y_from_p16 - ref).norm().item()
E_tail = err_norm / max(ref.norm().item(), 0.001 * np.sqrt(32.0))

result = {
    "min_p64": p64.min().item(),
    "num_p_below_fp16_min_normal": int((p64 < 6.104e-5).sum().item()),
    "num_p_below_fp16_min_subnormal": int((p64 < 5.96e-8).sum().item()),
    "num_p_snapped_to_zero_in_fp16": int((p16 == 0).sum().item()),
    "max_abs_p_quant_err": perr.max().item(),
    "sum_p_quant_err_abs": perr.sum().item(),
    "E_from_fp16_p_alone": E_tail,
    "err_norm_fp16_p_alone": err_norm,
    "ref_norm": ref.norm().item(),
    "E_budget": 0.001,
    "fp16_alone_exceeds_budget": E_tail > 0.001,
}
print(json.dumps(result))