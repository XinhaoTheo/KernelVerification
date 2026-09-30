import torch, json, sys, importlib.util

spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_07/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
gptq_matmul = m.gptq_matmul

torch.manual_seed(2)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev != 'cuda':
    print(json.dumps({"error": "no CUDA"})); sys.exit(0)

M, K, N, G = 16, 64, 40, 4   # N=40: multiple of 8, not of 32
g_idx = torch.randint(0, G, (K,), device=dev, dtype=torch.int32)
scales = (torch.rand(G, N, device=dev) * 0.1 + 0.01).float()
z_vals = torch.randint(1, 16, (G, N), device=dev, dtype=torch.int32)
q_vals = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)

def packK(x):
    Kd, Nd = x.shape
    p = torch.zeros((Kd // 8, Nd), device=x.device, dtype=torch.int32)
    for i in range(Kd):
        p[i // 8, :] |= (x[i, :] & 15) << ((i % 8) * 4)
    return p

def packN(x):
    Gd, Nd = x.shape
    p = torch.zeros((Gd, Nd // 8), device=x.device, dtype=torch.int32)
    for j in range(Nd):
        p[:, j // 8] |= (x[:, j] & 15) << ((j % 8) * 4)
    return p

b_packed = packK(q_vals)     # (8, 40)
zeros_packed = packN(z_vals) # (4, 5)
a = torch.randn(M, K, device=dev).half()

ar = torch.arange(N, device=dev)
ref = torch.zeros(M, N, device=dev)
for k in range(K):
    g = int(g_idx[k])
    zrow = ((zeros_packed[g, :].long() >> ((ar % 8) * 4)) & 15).float() + 1
    dq = (q_vals[k, :].float() - zrow) * scales[g]
    ref += a[:, k:k+1].float() * dq[None, :]

try:
    c = gptq_matmul(a, b_packed, scales, zeros_packed, g_idx, bits=4)
    torch.cuda.synchronize()
    err = (c - ref).abs().max().item()
    print(json.dumps({"metric": "max_abs_err for N=40 (N%32=8) vs kernel-convention reference",
                      "M": M, "K": K, "N": N,
                      "b_packed_shape": list(b_packed.shape),
                      "zeros_packed_shape": list(zeros_packed.shape),
                      "max_abs_err": err, "ref_magnitude": ref.abs().max().item(),
                      "note": "nonzero err or crash implies unmasked N-dim loads corrupt valid outputs (c3 confirmed); ~0 (fp rounding) implies OOB lanes do not affect stored outputs"}))
except Exception as e:
    print(json.dumps({"metric": "kernel_exception", "N": N, "error": repr(e),
                      "note": "crash on N%32!=0 supports c3 unmasked N-dimension loads"}))
