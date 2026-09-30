import torch, json, sys
sys.path.insert(0, '/root/cases/case_07')
from kernel import gptq_matmul

torch.manual_seed(0)
dev = 'cuda'
M, K, N = 64, 256, 128
bits, maxq, groupsize = 4, 15, 32
n_groups = K // groupsize

a = torch.randn(M, K, device=dev, dtype=torch.float16)
q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)
b_packed = torch.zeros((K // 8, N), device=dev, dtype=torch.int32)
for k in range(K):
    b_packed[k // 8, :] |= q[k, :] << ((k % 8) * bits)
scales = (torch.randn(n_groups, N, device=dev, dtype=torch.float16).abs() * 0.01 + 0.001)
zeros = torch.randint(0, 16, (n_groups, N // 8), device=dev, dtype=torch.int32)  # packed along N
g_idx = (torch.arange(K, device=dev) // groupsize).to(torch.int32)

c = gptq_matmul(a, b_packed, scales, zeros, g_idx, bits=4)

# reference: unpack zeros along N; effective zero = unpacked + 1 (AutoGPTQ)
n_ar = torch.arange(N, device=dev)
z_unpacked = ((zeros[:, n_ar // 8] >> ((n_ar % 8) * bits).unsqueeze(0)) & maxq).float()  # (n_groups, N)
q_f = q.float()
deq = torch.zeros(K, N, device=dev)
g_cpu = g_idx.cpu()
for k in range(K):
    g = int(g_cpu[k])
    deq[k] = (q_f[k] - (z_unpacked[g] + 1)) * scales[g].float()
ref = a.float() @ deq

err = (c - ref).abs()
print(json.dumps({
    "metric": "max abs error vs independent dequant reference (zero_eff = unpacked_zero + 1)",
    "max_abs_err": float(err.max()),
    "mean_abs_err": float(err.mean()),
    "shape": [M, N],
    "c_dtype": str(c.dtype),
    "allclose": bool(torch.allclose(c, ref, atol=1e-3, rtol=1e-3)),
}))
