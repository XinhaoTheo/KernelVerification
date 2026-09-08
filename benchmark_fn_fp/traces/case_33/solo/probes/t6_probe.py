
import torch, math, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_33/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

torch.manual_seed(0)
dev='cuda'
bits=4; maxq=15; ipw=32//bits

def ref(a, b_packed, scales, zeros, group_size, K, N):
    # unpack q
    q = torch.zeros((K,N), device=dev, dtype=torch.int32)
    for kk in range(K):
        word = b_packed[kk//ipw]
        q[kk] = (word >> ((kk % ipw)*bits)) & maxq
    z = torch.zeros((K,N), device=dev, dtype=torch.int32)
    for kk in range(K):
        g = kk//group_size
        zrow = zeros[g]
        for n in range(N):
            z[kk,n] = ((zrow[n//ipw].item() >> ((n % ipw)*bits)) & maxq) + 1
    s = torch.zeros((K,N), device=dev, dtype=torch.float32)
    for kk in range(K):
        s[kk] = scales[kk//group_size]
    w = (q.float()-z.float())*s
    return a @ w

def run(K, group_size, M=32, N=32):
    ng = math.ceil(K/group_size)
    a = torch.randn(M,K, device=dev, dtype=torch.float32)
    b_packed = torch.randint(-2**31, 2**31-1, (K//ipw, N), device=dev, dtype=torch.int32)
    scales = (torch.rand(ng, N, device=dev, dtype=torch.float32)*0.1+0.01).contiguous()
    zeros = torch.randint(-2**31, 2**31-1, (ng, N//ipw), device=dev, dtype=torch.int32)
    out = k.gptq_matmul(a, b_packed, scales, zeros, group_size, bits)
    r = ref(a, b_packed, scales, zeros, group_size, K, N)
    err = (out-r).abs()
    return dict(K=K, gs=group_size, num_groups=ng, max_abs=err.max().item(),
                ref_absmax=r.abs().max().item(),
                rel=(err.max()/(r.abs().max()+1e-9)).item())

res = {}
res['control_K64_gs32'] = run(64, 32)
res['partial_K48_gs32'] = run(48, 32)
res['partial_K96_gs64'] = run(96, 64)
# also show g_idx computed by wrapper for K=48,gs=32
import torch as t
K=48; gs=32; ngf=K//gs
gidx = t.clamp(t.arange(K)//gs, max=ngf-1)
res['g_idx_K48_gs32_unique'] = sorted(set(gidx.tolist()))
res['g_idx_correct_unique'] = sorted(set((t.arange(K)//gs).tolist()))
print(json.dumps(res, indent=2))
