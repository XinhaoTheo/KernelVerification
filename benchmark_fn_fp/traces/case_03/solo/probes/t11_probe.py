
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_03/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev='cuda'

def build(M,N,K,group_size,bits=4):
    ipw = 32//bits
    num_groups = (K + group_size - 1)//group_size
    g_idx = (torch.arange(K, device=dev)//group_size).to(torch.int32)
    q = torch.randint(0, 2**bits, (K,N), device=dev, dtype=torch.int32)
    packed = torch.zeros((K//ipw, N), device=dev, dtype=torch.int32)
    for k in range(K):
        packed[k//ipw] |= (q[k] << ((k % ipw)*bits))
    scales = (torch.rand((num_groups,N), device=dev)*0.1+0.01).float()
    zq = torch.randint(0, 2**bits, (num_groups,N), device=dev, dtype=torch.int32)
    qzeros = torch.zeros((num_groups, N//ipw), device=dev, dtype=torch.int32)
    for n in range(N):
        qzeros[:, n//ipw] |= (zq[:, n] << ((n % ipw)*bits))
    a = torch.randn((M,K), device=dev, dtype=torch.float32)
    z_exp = (zq[g_idx.long()] + 1).float()
    s_exp = scales[g_idx.long()]
    deq = (q.float() - z_exp) * s_exp
    ref = a @ deq
    return a, packed, scales, qzeros, g_idx, ref

out={}
cases = {
  "c2_K40_gs32":(32,32,40,32),
  "c2_K24_gs16":(32,32,24,16),
  "c2_K8_gs8":(32,32,8,8),
  "c2_K72_gs32":(32,32,72,32),
}
for name,(M,N,K,gs) in cases.items():
    try:
        a,packed,scales,qzeros,g_idx,ref = build(M,N,K,gs)
        c = m.gptq_matmul(a, packed, scales, qzeros, g_idx, bits=4)
        torch.cuda.synchronize()
        err = (c-ref).abs()
        out[name] = dict(K=K, gs=gs, status="ok",
                         max_abs=err.max().item(),
                         rel=(err.max()/ref.abs().max()).item(),
                         ref_absmax=ref.abs().max().item(),
                         frac_elems_off_1e3=(err>1e-3).float().mean().item())
    except Exception as e:
        out[name] = dict(K=K, gs=gs, status="exception", err=repr(e)[:300])
print("RESULT " + json.dumps(out))
