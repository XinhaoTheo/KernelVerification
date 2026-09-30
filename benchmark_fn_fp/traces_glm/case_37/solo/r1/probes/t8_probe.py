import json, importlib.util
import torch
import numpy as np

spec = importlib.util.spec_from_file_location("kernel_case_b", "/root/cases/case_b/kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Exact specified workload
x, A, B = mod.make_inputs(device="cuda")

# Reference computed BEFORE invoking the kernel, from the original float32 inputs, in float64.
# Two independent computations: torch fp64 and numpy fp64.
x64 = x.to(torch.float64); A64 = A.to(torch.float64); B64 = B.to(torch.float64)
ref = A64 @ x64 + B64 @ x64
xn = x.cpu().numpy().astype(np.float64)
An = A.cpu().numpy().astype(np.float64)
Bn = B.cpu().numpy().astype(np.float64)
ref_np = An @ xn + Bn @ xn

out1 = mod.run(x, A, B)
out2 = mod.run(x, A, B)
torch.cuda.synchronize()

o64 = out1.to(torch.float64)
den_t = max(torch.norm(ref).item(), 0.001 * np.sqrt(64))
E_torch = torch.norm(o64 - ref).item() / den_t
den_np = max(np.linalg.norm(ref_np), 0.001 * np.sqrt(64))
E_np = float(np.linalg.norm(o64.cpu().numpy() - ref_np) / den_np)
E_run2 = torch.norm(out2.to(torch.float64) - ref).item() / den_t
deterministic = bool(torch.equal(out1, out2))
finite = bool(torch.isfinite(out1).all().item())

# Mutation check: regenerate inputs from the fixed seed and compare with post-run tensors
x_fresh, A_fresh, B_fresh = mod.make_inputs(device="cuda")
inputs_unmutated = bool(torch.equal(x, x_fresh) and torch.equal(A, A_fresh) and torch.equal(B, B_fresh))
ref_agree = float(np.abs(ref.cpu().numpy() - ref_np).max())

# Error attribution: per-branch quantization error vectors, norms, and correlation
ref_a = A64 @ x64; ref_b = B64 @ x64
sa = A64.abs().amax(dim=1) / 7.0
qa = torch.clamp(torch.floor(A64 / sa[:, None] + 0.5), -7.0, 7.0)
sb = B64.abs().amax(dim=1) / 7.0
qb = torch.clamp(torch.floor(B64 / sb[:, None] + 0.5), -7.0, 7.0)
eA = (qa * sa[:, None]) @ x64 - ref_a
eB = (qb * sb[:, None]) @ x64 - ref_b
corr = float((torch.dot(eA, eB) / (eA.norm() * eB.norm())).item())

result = {
    "metric": "contract E = ||out - (A@x + B@x)_fp64||_2 / max(||ref||_2, 0.001*sqrt(64)) on run(*make_inputs())",
    "metric_reason": "Independent recomputation of the decisive metric: reference built BEFORE kernel invocation via torch fp64 and a separate numpy fp64 matmul, plus determinism and input-mutation checks, to rule out probe artifacts in the first measurement (t6).",
    "E_torch_fp64": E_torch,
    "E_numpy_fp64_independent": E_np,
    "E_second_run": E_run2,
    "all_E_le_0.1": bool(E_torch <= 0.1 and E_np <= 0.1 and E_run2 <= 0.1),
    "all_outputs_finite": finite,
    "deterministic_across_runs": deterministic,
    "inputs_unmutated_by_run": inputs_unmutated,
    "torch_vs_numpy_ref_max_abs_diff": ref_agree,
    "numerator_l2": torch.norm(o64 - ref).item(),
    "denominator": den_t,
    "branch_A_error_norm": eA.norm().item(),
    "branch_B_error_norm": eB.norm().item(),
    "combined_error_norm": (eA + eB).norm().item(),
    "branch_error_correlation": corr,
    "seed_used": mod.SEED,
    "verdict_signal": "COMPLIANT" if (E_torch <= 0.1 and E_np <= 0.1 and finite) else "VIOLATION",
}
print(json.dumps(result))
