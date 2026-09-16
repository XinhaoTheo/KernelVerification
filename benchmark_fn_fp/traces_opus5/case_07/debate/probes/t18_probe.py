
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_07/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
dev = 'cuda'
M, N, K, gs = 64, 64, 64, 32   # K%16==0 and N%32==0 so the c2 K/N-tail defect cannot confound
G = K // gs

a = torch.randn(M, K, device=dev, dtype=torch.float32)
q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int32)     # unpacked 4-bit weights
v = torch.randint(0, 15, (G, N), device=dev, dtype=torch.int32)     # the value actually STORED in the zeros tensor
scales = torch.rand(G, N, device=dev, dtype=torch.float32) * 0.1 + 0.01
g_idx = (torch.arange(K, device=dev) // gs).to(torch.int32)

b_packed = torch.zeros(K // 8, N, device=dev, dtype=torch.int32)
for row in range(K // 8):
    for i in range(8):
        b_packed[row] |= (q[row * 8 + i] & 0xF) << (4 * i)

zp = torch.zeros(G, N // 8, device=dev, dtype=torch.int32)
for col in range(N // 8):
    for i in range(8):
        zp[:, col] |= (v[:, col * 8 + i] & 0xF) << (4 * i)

def ref(zero_vals):
    w = (q.double() - zero_vals.double()[g_idx]) * scales.double()[g_idx]
    return a.double() @ w

# Convention A: literal problem.txt formula -> zero == stored value v
ref_literal = ref(v)
# Convention B: AutoGPTQ storage -> true zero == stored value + 1
ref_plus1 = ref(v + 1)

c = k.gptq_matmul(a, b_packed, scales, zp, g_idx, bits=4)
torch.cuda.synchronize()

def e(o, r):
    d = (o.double() - r).abs()
    return float(d.max()), float((d / (r.abs() + 1e-9)).max())

lit_abs, lit_rel = e(c, ref_literal)
p1_abs, p1_rel = e(c, ref_plus1)
# predicted bias if the kernel wrongly adds 1: -scale[g,n]*sum_{k in g} a[m,k]
bias = (a.double() @ scales.double()[g_idx])
diff_refs = (ref_literal - ref_plus1).abs()

print(json.dumps(dict(
    shape=[M, N, K], group_size=gs,
    err_vs_literal_zero_ref_maxabs=lit_abs, err_vs_literal_zero_ref_maxrel=lit_rel,
    err_vs_zero_plus1_ref_maxabs=p1_abs, err_vs_zero_plus1_ref_maxrel=p1_rel,
    ref_absmax=float(ref_literal.abs().max()),
    predicted_bias_maxabs=float(bias.abs().max()),
    actual_gap_between_two_refs_maxabs=float(diff_refs.max()),
    kernel_matches=("zero_plus1_autogptq" if p1_abs < lit_abs else "literal_zero"),
)))
