
import torch, json, sys

sys.path.insert(0, '/root/cases/case_07')
try:
    from kernel import gptq_matmul
except Exception as e:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_07/kernel.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        gptq_matmul = m.gptq_matmul
    except Exception as e2:
        print(json.dumps({"error": str(e2)})); sys.exit(0)

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
M, K, N = 16, 64, 64
num_groups = 4
g_idx = torch.randint(0, num_groups, (K,), device=dev, dtype=torch.int32)

# random scales and nonzero zero-points
scales = (torch.rand(num_groups, N, device=dev) * 0.1 + 0.01).float()
z_vals = torch.randint(1, 16, (num_groups, N), device=dev, dtype=torch.int32)  # nonzero zeros
q_vals = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)

def pack(x, bits=4):
    vals_per_word = 32 // bits
    Kd, Nd = x.shape
    packed = torch.zeros((Kd // vals_per_word, Nd), device=x.device, dtype=torch.int32)
    for i in range(Kd):
        packed[i // vals_per_word, :] |= (x[i, :] & 15) << ((i % vals_per_word) * bits)
    return packed

b_packed = pack(q_vals)
zeros_packed = pack(z_vals)

a = torch.randn(M, K, device=dev).half() if dev=='cuda' else torch.randn(M, K).float()

# reference per contract formula: (q - zero) * scale, zero unpacked raw (NO +1)
ref = torch.zeros(M, N, device=dev)
for k in range(K):
    g = int(g_idx[k]); z = int(z_vals[g])  # z_vals indexed [group, n]
    # unpack zero word
    zero_word = zeros_packed[g, :].long()
    zrow = (zero_word >> ((torch.arange(N, device=dev) % 8) * 4)) & 15
    dq = (q_vals[k, :].float() - zrow.float()) * scales[g]
    ref[:, :] += a[:, k:k+1].float() * dq[None, :]

try:
    c = gptq_matmul(a, b_packed, scales, zeros_packed, g_idx, bits=4)
    max_abs = (c - ref).abs().max().item()
    # also compare against GPTQ +1 convention
    ref_plus1 = torch.zeros(M, N, device=dev)
    for k in range(K):
        g = int(g_idx[k])
        zero_word = zeros_packed[g, :].long()
        zrow = (zero_word >> ((torch.arange(N, device=dev) % 8) * 4)) & 15
        dq = (q_vals[k, :].float() - (zrow.float() + 1)) * scales[g]
        ref_plus1[:, :] += a[:, k:k+1].float() * dq[None, :]
    max_abs_plus1 = (c - ref_plus1).abs().max().item()
    print(json.dumps({"metric": "max_abs_err vs contract (q-zero)*scale and vs (q-zero-1)*scale",
                      "max_abs_err_contract": max_abs,
                      "max_abs_err_plus1_convention": max_abs_plus1,
                      "M": M, "K": K, "N": N, "bits": 4,
                      "note": "if kernel matches +1 convention but not contract formula, +1 offset confirmed"}))
except Exception as e:
    print(json.dumps({"error": repr(e)}))
