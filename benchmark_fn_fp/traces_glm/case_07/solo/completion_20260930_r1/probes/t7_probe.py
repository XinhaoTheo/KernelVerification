
import torch, json, sys
sys.path.insert(0, '/root/cases/case_07')
from kernel import gptq_matmul

torch.manual_seed(0)
dev = 'cuda'
M, K, N = 64, 256, 128
bits = 4
maxq = 15
groupsize = 32
n_groups = K // groupsize

a = torch.randn(M, K, device=dev, dtype=torch.float16)
q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)
# pack 8 values per int32 along K (little-endian nibbles)
b_packed = torch.zeros((K // 8, N), device=dev, dtype=torch.int32)
for k in range(K):
    word, off = k // 8, k % 8
    b_packed[word, :] |= q[k, :] << (off * bits)
scales = (torch.randn(n_groups, N, device=dev, dtype=torch.float16).abs() * 0.01 + 0.001)
zeros = torch.randint(0, 16, (n_groups, N), device=dev, dtype=torch.int32)  # packed zeros along N
g_idx = (torch.arange(K, device=dev) // groupsize).to(torch.int32)

c = gptq_matmul(a, b_packed, scales, zeros, g_idx, bits=4)

# reference: unpack zeros along N (8 per word), effective zero = unpacked + 1 (AutoGPTQ convention)
q_f = q.float()
zero_eff = torch.zeros_like(q_f)
for n in range(N):
    word, off = n // 8, n % 8
    z_un = ((zeros[word, :] >> (off * bits)) & maxq).float()  # per column? careful
# do it vectorized: zero value for column n of group g
z_unpacked = ((zeros[:, (torch.arange(N, device=dev) // 8)] >> ((torch.arange(N, device=dev) % 8) * bits).unsqueeze(0)) & maxq).float()  # (n_groups, N)
deq = torch.zeros(K, N, device=dev)
for k in range(K):
    g = g_idx[k].item()
    deq[k] = (q_f[k] - (z_unpacked[g] + 1)) * scales[g].float()
ref = (a.float() @ deq)

err = (c - ref).abs()
print(json.dumps({
    "metric": "max abs error vs independent dequant reference (zero_eff = unpacked_zero + 1)",
    "max_abs_err": float(err.max()),
    "mean_abs_err": float(err.mean()),
    "shape": [M, N],
    "c_dtype": str(c.dtype),
    "allclose": bool(torch.allclose(c, ref, atol=1e-2, rtol=1e-2)),
}))
