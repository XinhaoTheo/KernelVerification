
import json, torch, sys
sys.path.insert(0, "/root/cases/case_07")
import importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_07/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
dev='cuda'
M,N,K = 64,64,64
gs = 32                     # group size along K
G = K//gs
bits=4; maxq=15

a = torch.randn(M,K, device=dev, dtype=torch.float32)
q = torch.randint(0,16,(K,N), device=dev, dtype=torch.int32)     # unpacked 4-bit weights
z = torch.randint(0,16,(G,N), device=dev, dtype=torch.int32)     # literal zero-points (as in problem.txt)
scales = (torch.rand(G,N, device=dev, dtype=torch.float32)*0.1+0.01)
g_idx = (torch.arange(K, device=dev)//gs).to(torch.int32)

# pack q: 8 nibbles per int32 along K
b_packed = torch.zeros(K//8, N, device=dev, dtype=torch.int32)
for i in range(8):
    b_packed |= (q[i::8*0+ i:0] if False else q[i:K:8]*0)  # placeholder, replaced below
b_packed = torch.zeros(K//8, N, device=dev, dtype=torch.int32)
for row in range(K//8):
    for i in range(8):
        b_packed[row] |= (q[row*8+i] & 0xF) << (4*i)

def pack_zeros(zz):
    p = torch.zeros(G, N//8, device=dev, dtype=torch.int32)
    for col in range(N//8):
        for i in range(8):
            p[:,col] |= (zz[:, col*8+i] & 0xF) << (4*i)
    return p

# Convention A: harness stores literal zero (problem.txt formula)
zeros_literal = pack_zeros(z)
# Convention B: harness stores zero-1 (AutoGPTQ); use z-1 clamped to >=0
z_minus1 = (z.clamp(min=1) - 1)
zeros_autogptq = pack_zeros(z_minus1)

def ref(qq, zz):
    # dequant per problem.txt literal formula
    w = (qq.float() - zz[g_idx].float()) * scales[g_idx].float()
    return a.double() @ w.double()

out_lit = k.gptq_matmul(a, b_packed, scales, zeros_literal, g_idx, bits=4)
torch.cuda.synchronize()
# reference under literal packing, literal formula (zero = z)
r_lit = ref(q, z)
# reference matching kernel's internal assumption when packed value = z: subtract z+1
r_lit_plus1 = ref(q, z+1)

out_ag = k.gptq_matmul(a, b_packed, scales, zeros_autogptq, g_idx, bits=4)
torch.cuda.synchronize()
# under AutoGPTQ packing (stored z-1), literal-formula reference with true zero-point z
r_ag_true = ref(q, z_minus1+1)

def e(o, r):
    d = (o.double()-r).abs()
    return float(d.max().item()), float((d/(r.abs()+1e-9)).max().item())

m1 = e(out_lit, r_lit)
m2 = e(out_lit, r_lit_plus1)
m3 = e(out_ag, r_ag_true)
res = dict(
  shape=[M,N,K], group_size=gs,
  err_literalpack_vs_literalformula_maxabs=m1[0],
  err_literalpack_vs_zeroplus1_maxabs=m2[0],
  err_autogptqpack_vs_truezero_maxabs=m3[0],
  ref_absmax=float(r_lit.abs().max().item()),
  bias_predicted_maxabs=float((a.double() @ scales[g_idx].double()).abs().max().item()),
)
print(json.dumps(res))
