
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_07/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

dev='cuda'
M,K,N,gs = 32,64,64,32
torch.manual_seed(0)
groups=K//gs
q = torch.randint(0,16,(K,N),device=dev,dtype=torch.int64)
zp = torch.randint(0,15,(groups,N),device=dev,dtype=torch.int64)  # keep zp+1 <=15
scales = (torch.rand((groups,N),device=dev)*0.02+0.005).float()
g_idx = (torch.arange(K,device=dev)//gs).to(torch.int32)

def wrap32(x):
    x = x & 0xFFFFFFFF
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)

qw = torch.zeros((K//8,N),device=dev,dtype=torch.int64)
for k in range(K):
    qw[k//8] |= (q[k] << ((k%8)*4))
qweight = wrap32(qw)
qz = torch.zeros((groups,N//8),device=dev,dtype=torch.int64)
for n in range(N):
    qz[:,n//8] |= (zp[:,n] << ((n%8)*4))
qzeros = wrap32(qz)

a = torch.randn((M,K),device=dev,dtype=torch.float32)
gl = g_idx.long()
deq_plus = (q.double() - (zp+1)[gl].double()) * scales[gl].double()
deq_bare = (q.double() - zp[gl].double()) * scales[gl].double()
ref_plus = (a.double() @ deq_plus)
ref_bare = (a.double() @ deq_bare)

c = kern.gptq_matmul(a, qweight, scales, qzeros, g_idx, bits=4).double()
den_p = ref_plus.abs().max().item(); den_b = ref_bare.abs().max().item()
e_p = (c-ref_plus).abs().max().item(); e_b = (c-ref_bare).abs().max().item()
# predicted systematic offset between conventions
off = (a.double() @ scales[gl].double()).abs().max().item()
print(json.dumps({
 "shape":[M,K,N],"group_size":gs,
 "max_abs_err_vs_zp_plus1_ref":e_p,
 "max_abs_err_vs_bare_zp_ref":e_b,
 "ref_plus_max_abs":den_p,"ref_bare_max_abs":den_b,
 "max_rel_err_vs_zp_plus1":e_p/den_p,
 "max_rel_err_vs_bare_zp":e_b/den_b,
 "predicted_convention_offset_max_abs":off,
 "kernel_matches":"zp_plus1" if e_p<e_b else "bare_zp"}))
