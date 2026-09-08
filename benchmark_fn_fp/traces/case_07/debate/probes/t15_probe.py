
import torch, importlib.util, json, traceback
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_07/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)
dev='cuda'
def wrap32(x):
    x = x & 0xFFFFFFFF
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)

def run(M,K,N,gs,actorder=False,seed=7):
    torch.manual_seed(seed)
    groups=K//gs
    q = torch.randint(0,16,(K,N),device=dev,dtype=torch.int64)
    zp = torch.randint(0,15,(groups,N),device=dev,dtype=torch.int64)
    scales = (torch.rand((groups,N),device=dev)*0.02+0.005).float()
    if actorder:
        g_idx = torch.randint(0,groups,(K,),device=dev).to(torch.int32)
    else:
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
    return {"max_abs_err":e,"ref_max_abs":d,"max_rel_err":e/d,"finite":bool(torch.isfinite(c).all().item())}

cases = {
 "M32_K64_N64_g32": (32,64,64,32,False),
 "M64_K128_N64_g64": (64,128,64,64,False),
 "M128_K256_N128_g128": (128,256,128,128,False),
 "M33_K64_N64_g32_Mtail": (33,64,64,32,False),
 "M32_K64_N96_g32": (32,64,96,32,False),
 "M32_K64_N40_g32_Ntail": (32,64,40,32,False),
 "M64_K128_N64_g32_actorder": (64,128,64,32,True),
 "M32_K512_N64_g128_deepK": (32,512,64,128,False),
}
out={}
for name,(M,K,N,gs,ao) in cases.items():
    try:
        out[name]=run(M,K,N,gs,ao)
    except Exception as ex:
        out[name]={"error":repr(ex)[:200]}
worst=max([v["max_rel_err"] for v in out.values() if "max_rel_err" in v], default=None)
print(json.dumps({"cases":out,"worst_max_rel_err":worst}))
