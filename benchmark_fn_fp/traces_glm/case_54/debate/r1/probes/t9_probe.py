import torch, json, sys
sys.path.insert(0, "/root/cases/case_127")
import kernel as K

dev = "cuda" if torch.cuda.is_available() else "cpu"
bits = torch.arange(65536, dtype=torch.int32)
fp16_all = torch.tensor(bits.view(torch.uint8).numpy().copy(), dtype=torch.uint8).view(torch.float16)
x_all = fp16_all[torch.isfinite(fp16_all)].to(dev)
out = K.run(x_all)

# truncated formula evaluated in float64 (exact-formula baseline)
m = x_all.to(torch.float64).abs()
t = 1.0 / (1.0 + 0.3275911 * m)
p = 1.06141*t - 1.45315
p = p*t + 1.42141
p = p*t - 0.28450
p = p*t + 0.25483
p = p*t
val = 1.0 - p*torch.exp(-(m*m))
val = torch.where(x_all.to(torch.float64) < 0, -val, val)

drift = (out.to(torch.float64) - val).abs()
err_formula = (val - torch.erf(x_all.to(torch.float64))).abs()
err_total = (out.to(torch.float64) - torch.erf(x_all.to(torch.float64))).abs()
i = int(drift.argmax())
print(json.dumps({
    "max_fp32_eval_drift": float(drift.max()),
    "drift_worst_x": float(x_all[i]),
    "max_truncated_formula_err_exact_arith": float(err_formula.max()),
    "max_total_err_kernel_vs_erf": float(err_total.max()),
    "num_drift_above_1e_6": int((drift > 1e-6).sum()),
    "tolerance": 1e-6,
    "out_finite": bool(torch.isfinite(out).all().item()),
}))
