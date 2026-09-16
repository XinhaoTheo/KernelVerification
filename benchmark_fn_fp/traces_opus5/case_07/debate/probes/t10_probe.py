
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_07/kernel.py")
kern = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern)

dev='cuda'
M,K,N,gs = 64,256,64,64
torch.manual_seed(1)
groups=K//gs
q = torch.randint(0,16,(K,N),device=dev,dtype=torch.int64)
zp = torch.randint(0,15,(groups,N),device=dev,dtype=torch.int64)
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
gl=g_idx.long()
deq32 = ((q.float() - (zp+1)[gl].float()) * scales[gl]).float()
ref64 = a.double() @ deq32.double()
den = ref64.abs().max().item()

c = kern.gptq_matmul(a, qweight, scales, qzeros, g_idx, bits=4).double()
err_k = (c-ref64).abs().max().item()

torch.backends.cuda.matmul.allow_tf32=False
t_fp32 = (a@deq32).double()
err_fp32 = (t_fp32-ref64).abs().max().item()
torch.backends.cuda.matmul.allow_tf32=True
t_tf32 = (a@deq32).double()
err_tf32 = (t_tf32-ref64).abs().max().item()
torch.backends.cuda.matmul.allow_tf32=False

print(json.dumps({
 "device":torch.cuda.get_device_name(0),
 "shape":[M,K,N],
 "kernel_max_abs_err_vs_fp64":err_k,"kernel_max_rel_err":err_k/den,
 "torch_fp32_max_abs_err":err_fp32,"torch_fp32_max_rel_err":err_fp32/den,
 "torch_tf32_max_abs_err":err_tf32,"torch_tf32_max_rel_err":err_tf32/den,
 "ref_max_abs":den,
 "kernel_closer_to":"tf32" if abs(err_k-err_tf32)<abs(err_k-err_fp32) else "fp32"}))
