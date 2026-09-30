import torch, sys, json
sys.path.insert(0, "/root/cases/case_16")
from kernel import matmul

M, N, K = 65, 65, 33
torch.manual_seed(0)
a = torch.randn(M, K, device="cuda", dtype=torch.float32)
b = torch.randn(K, N, device="cuda", dtype=torch.float32)
c = matmul(a, b)
ref = torch.matmul(a, b)
# expected value if tail dropped: sum over first 32 k
ref_dropped = a[:, :32] @ b[:32, :]
err_vs_ref = (c - ref).abs().max().item()
err_vs_dropped = (c - ref_dropped).abs().max().item()
print(json.dumps({
    "M": M, "N": N, "K": K,
    "max_abs_err_vs_torch_matmul": err_vs_ref,
    "max_abs_err_vs_first32_truncated": err_vs_dropped,
    "tail_dropped_hypothesis_confirmed": err_vs_ref > 1e-3 and err_vs_dropped < 1e-3
}))