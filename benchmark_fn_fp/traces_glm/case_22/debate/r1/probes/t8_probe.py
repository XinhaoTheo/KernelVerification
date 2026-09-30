import torch, json
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_22/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
K = 8 * 1024 * 1024  # 8M elements, cancellation-heavy
# construct b so that the true dot is tiny while partial sums are large
a = (torch.randn(K, device="cuda", dtype=torch.float32) * 1e3)
b = torch.randn(K, device="cuda", dtype=torch.float32)
b -= (a @ b) / (a @ a) * a  # orthogonalize so exact dot = 0-ish
ref64 = torch.dot(a.double(), b.double()).item()

vals = [k.splitk_dot(a, b).item() for _ in range(50)]
import statistics
spread = max(vals) - min(vals)
max_abs_err = max(abs(v - ref64) for v in vals)
# measure magnitude of partial sums for scale
partial_scale = (a * b).abs().sum().item()
res = {
    "K": K, "split": 512,
    "reference_dot_fp64": ref64,
    "kernel_min": min(vals), "kernel_max": max(vals),
    "run_to_run_spread": spread,
    "max_abs_err_vs_ref": max_abs_err,
    "max_rel_err_vs_partial_scale": max_abs_err / partial_scale,
    "max_rel_err_vs_ref": (max_abs_err / abs(ref64)) if ref64 != 0 else None,
}
print(json.dumps(res))