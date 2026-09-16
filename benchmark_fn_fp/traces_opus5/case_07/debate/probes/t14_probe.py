
import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_07/kernel.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
torch.manual_seed(0)
dev='cuda'

def build(K, M=32, N=32, gs=8):
    G = K//gs
    a = torch.randn(M,K, device=dev, dtype=torch.float32)
    q = torch.randint(0,16,(K,N), device=dev, dtype=torch.int32)
    zst = torch.randint(0,15,(G,N), device=dev, dtype=torch.int32)  # stored zeros (AutoGPTQ: true zero = stored+1)
    scales = torch.rand(G,N, device=dev, dtype=torch.float32)*0.1+0.01
    g_idx = (torch.arange(K, device=dev)//gs).to(torch.int32)
    b_packed = torch.zeros(K//8, N, device=dev, dtype=torch.int32)
    for row in range(K//8):
        for i in range(8):
            b_packed[row] |= (q[row*8+i] & 0xF) << (4*i)
    zp = torch.zeros(G, N//8, device=dev, dtype=torch.int32)
    for col in range(N//8):
        for i in range(8):
            zp[:,col] |= (zst[:, col*8+i] & 0xF) << (4*i)
    w = (q.float() - (zst[g_idx].float()+1)) * scales[g_idx].float()
    ref = a.double() @ w.double()
    return a, b_packed, scales, zp, g_idx, ref

out = {}
for K in (32, 24, 40, 8):
    try:
        a,bp,sc,zp,gi,ref = build(K)
        c = k.gptq_matmul(a, bp, sc, zp, gi, bits=4)
        torch.cuda.synchronize()
        d = (c.double()-ref).abs()
        out[f"K{K}"] = dict(maxabs=float(d.max()), maxrel=float((d/(ref.abs()+1e-9)).max()),
                            ref_absmax=float(ref.abs().max()), faulted=False)
    except Exception as ex:
        out[f"K{K}"] = dict(faulted=True, error=repr(ex)[:300])
print(json.dumps(out))
