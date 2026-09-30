
import torch, json, sys, importlib.util

spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_07/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
gptq_matmul = m.gptq_matmul

torch.manual_seed(2)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev != 'cuda':
    print(json.dumps({"error": "no CUDA, cannot run triton kernel"})); sys.exit(0)

M, K, N = 16, 64, 40   # N = 40: multiple of 8 (zeros word n//8=5 valid up to 39) but not multiple of 32
num_groups = 4
g_idx = torch.randint(0, num_groups, (K,), device=dev, dtype=torch.int32)
scales = (torch.rand(num_groups, N, device=dev) * 0.1 + 0.01).float()
z_vals = torch.randint(0, 16, (num_groups, N), device=dev, dtype=torch.int32)
q_vals = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)

def pack(x, bits=4):
    vpw = 32 // bits
    Kd, Nd = x.shape
    packed = torch.zeros((Kd // vpw, Nd), device=x.device, dtype=torch.int32)
    for i in range(Kd):
        packed[i // vpw, :] |= (x[i, :] & 15) << ((i % vpw) * bits)
    return packed

b_packed = pack(q_vals)
zeros_packed = pack(z_vals)
a = torch.randn(M, K, device=dev).half()

def reference(use_plus1):
    ref = torch.zeros(M, N, device=dev)
    ar = torch.arange(N, device=dev)
    for k in range(K):
        g = int(g_idx[k])
        zrow = ((zeros_packed[g, :].long() >> ((ar % 8) * 4)) & 15).float()
        if use_plus1: zrow = zrow + 1
        dq = (q_vals[k, :].float() - zrow) * scales[g]
        ref += a[:, k:k+1].float() * dq[None, :]
    return ref

ref1 = reference(True)  # kernel convention (zeros+1)
try:
    c = gptq_matmul(a, b_packed, scales, zeros_packed, g_idx, bits=4)
    torch.cuda.synchronize()
    err1 = (c - ref1).abs().max().item()
    print(json.dumps({"metric": "max_abs_err for N=40 (N%32=8) vs kernel-convention reference",
                      "M": M, "K": K, "N": N,
                      "max_abs_err": err1,
                      "note": "error or crash implies unmasked N-dim loads corrupt valid outputs (c3 confirmed); ~0 implies OOB reads happen but do not affect stored lanes"}))
except Exception as e:
    print(json.dumps({"metric": "kernel_exception", "N": N, "error": repr(e),
                      "note": "crash on N%32!=0 supports c3 out-of-bounds load"}))
