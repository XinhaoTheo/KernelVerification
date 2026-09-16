
import torch, importlib.util, json, traceback
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_07/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev='cuda'
def wrap32(x):
    x = x & 0xFFFFFFFF
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)

def run(K, gs, seed=2, M=32, N=64):
    torch.manual_seed(seed)
    groups=K//gs
    q = torch.randint(0,16,(K,N),device=dev,dtype=torch.int64)
    zp = torch.randint(0,15,(groups,N),device=dev,dtype=torch.int64)
    scales = (torch.rand((groups,N),device=dev)*0.02+0.005).float()
    g_idx = (torch.arange(K,device=dev)//gs).to(torch.int32)
    qw = torch.zeros((K//8,N),device=dev,dtype=torch.int64)
    for k in range(K):
        qw[k//8] |= (q[k] << ((k%8)*4))
    qweight = wrap32(qw)
    qz = torch.zeros((groups,N//8),device=dev,dtype=torch.int64)
    for n in range(N):
        qz[:,n//8] |= (zp[:,n] << ((n%8)*4))
    qzeros = wrap32(qz)
    a = torch.randn((M,K),device=dev,dtype=torch.float32)
    gl=g_idx.long()
    deq = (q.double() - (zp+1)[gl].double()) * scales[gl].double()
    ref = a.double() @ deq
    c = kern.gptq_matmul(a, qweight, scales, qzeros, g_idx, bits=4).double()
    torch.cuda.synchronize()
    e=(c-ref).abs().max().item(); d=ref.abs().max().item()
    return {"K":K,"group_size":gs,"max_abs_err":e,"ref_max_abs":d,"max_rel_err":e/d}

out={}
try:
    out["control_K32"]=run(32,8)
except Exception as ex:
    out["control_K32"]={"error":repr(ex)}
try:
    out["tail_K24"]=run(24,8)
except Exception as ex:
    out["tail_K24"]={"error":repr(ex),"tb":traceback.format_exc()[-400:]}
print(json.dumps(out))
