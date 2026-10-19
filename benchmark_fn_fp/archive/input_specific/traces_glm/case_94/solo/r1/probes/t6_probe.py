import json, importlib.util
import torch
import numpy as np

spec = importlib.util.spec_from_file_location("kernel_case_b", "/root/cases/case_b/kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Exact specified workload
x, A, B = mod.make_inputs(device="cuda")
out = mod.run(x, A, B)
torch.cuda.synchronize()

# Reference: A @ x + B @ x on the ORIGINAL float32 inputs, evaluated in float64
x64 = x.to(torch.float64)
A64 = A.to(torch.float64)
B64 = B.to(torch.float64)
ref = A64 @ x64 + B64 @ x64
ref_a = A64 @ x64
ref_b = B64 @ x64

out64 = out.to(torch.float64)
num = torch.norm(out64 - ref).item()
den = max(torch.norm(ref).item(), 0.001 * np.sqrt(64))
E = num / den
finite = bool(torch.isfinite(out).all().item())

# Diagnostic: emulate the kernel's per-row +/-7 quantization in float64 to attribute error
sa = A64.abs().amax(dim=1) / 7.0
sb = B64.abs().amax(dim=1) / 7.0
qa = torch.clamp(torch.floor(A64 / sa[:, None] + 0.5), -7.0, 7.0)
qb = torch.clamp(torch.floor(B64 / sb[:, None] + 0.5), -7.0, 7.0)
ya = (qa * sa[:, None]) @ x64
yb = (qb * sb[:, None]) @ x64
Ea = torch.norm(ya - ref_a).item() / max(torch.norm(ref_a).item(), 1e-12)
Eb = torch.norm(yb - ref_b).item() / max(torch.norm(ref_b).item(), 1e-12)
emul_max_dev = (out64 - (ya + yb)).abs().max().item()

result = {
    "metric": "contract E = ||out - (A@x + B@x)_fp64||_2 / max(||ref||_2, 0.001*sqrt(64)) on run(*make_inputs())",
    "metric_reason": "The contract declares exactly this normalized L2 metric against a float64 reference on the original float32 inputs; the bound is E <= 0.1 with all outputs finite.",
    "E": E,
    "E_le_0.1": bool(E <= 0.1),
    "numerator_l2": num,
    "denominator": den,
    "ref_norm": torch.norm(ref).item(),
    "all_outputs_finite": finite,
    "out_shape": list(out.shape),
    "out_dtype": str(out.dtype),
    "input_shapes": [list(x.shape), list(A.shape), list(B.shape)],
    "input_dtypes": [str(x.dtype), str(A.dtype), str(B.dtype)],
    "seed_used": mod.SEED,
    "diag_branch_E_A": Ea,
    "diag_branch_E_B": Eb,
    "diag_gpu_vs_cpu_emulation_max_abs": emul_max_dev,
    "verdict_signal": "VIOLATION" if (E > 0.1 or not finite) else "COMPLIANT",
}
print(json.dumps(result))
