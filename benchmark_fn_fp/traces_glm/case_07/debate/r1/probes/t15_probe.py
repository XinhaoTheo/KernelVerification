import torch, json, sys, importlib.util

spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_07/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
gptq_matmul = m.gptq_matmul

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev != 'cuda':
    print(json.dumps({"error": "no CUDA"})); sys.exit(0)

M, K, N, G = 16, 64, 64, 4
g_idx = torch.randint(0, G, (K,), device=dev, dtype=torch.int32)
scales = (torch.rand(G, N, device=dev) * 0.1 + 0.01).float()
z_vals = torch.randint(1, 16, (G, N), device=dev, dtype=torch.int32)  # nonzero zeros
q_vals = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)

def packK(x):  # (K,N) -> (K/8, N), pack along dim0
    Kd, Nd = x.shape
    p = torch.zeros((Kd // 8, Nd), device=x.device, dtype=torch.int32)
    for i in range(Kd):
        p[i // 8, :] |= (x[i, :] & 15) << ((i % 8) * 4)
    return p

def packN(x):  # (G,N) -> (G, N/8), pack along dim1
    Gd, Nd = x.shape
    p = torch.zeros((Gd, Nd // 8), device=x.device, dtype=torch.int32)
    for j in range(Nd):
        p[:, j // 8] |= (x[:, j] & 15) << ((j % 8) * 4)
    return p

b_packed = packK(q_vals)
zeros_packed = packN(z_vals)
a = torch.randn(M, K, device=dev).half()

ar = torch.arange(N, device=dev)
def reference(use_plus1):
    ref = torch.zeros(M, N, device=dev)
    for k in range(K):
        g = int(g_idx[k])
        zrow = ((zeros_packed[g, :].long() >> ((ar % 8) * 4)) & 15).float()
        if use_plus1: zrow = zrow + 1
        dq = (q_vals[k, :].float() - zrow) * scales[g]
        ref += a[:, k:k+1].float() * dq[None, :]
    return ref

ref0, ref1 = reference(False), reference(True)
c = gptq_matmul(a, b_packed, scales, zeros_packed, g_idx, bits=4)
torch.cuda.synchronize()
err0 = (c - ref0).abs().max().item()
err1 = (c - ref1).abs().max().item()
scale_ref = ref0.abs().max().item()
print(json.dumps({"metric": "max_abs_err vs contract formula (q-zero)*scale vs GPTQ (q-zero-1)*scale",
                  "M": M, "K": K, "N": N, "bits": 4,
                  "max_abs_err_contract_formula": err0,
                  "max_abs_err_plus1_convention": err1,
                  "ref_magnitude": scale_ref,
                  "note": "if err1<<err0, kernel uses zero+1 and deviates from the stated contract formula (c1 confirmed); if err0 small, c1 rebutted"}))
