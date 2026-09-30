import torch, sys, json
sys.path.insert(0, "/root/cases/case_16")
from kernel import matmul

M, N, K = 65, 65, 16
torch.manual_seed(0)
a = torch.randn(M, K, device="cuda", dtype=torch.float32)
b = torch.randn(K, N, device="cuda", dtype=torch.float32)
c = matmul(a, b)
ref = torch.matmul(a, b)
err = (c - ref).abs().max().item()
all_zero = bool((c == 0).all().item())
print(json.dumps({
    "M": M, "N": N, "K": K,
    "max_abs_err_vs_torch_matmul": err,
    "output_all_zeros": all_zero,
    "loop_never_ran_confirmed": err > 1e-3 or all_zero
}))