import torch, json, sys
import triton
sys.path.insert(0,'/root/cases/case_33')
from kernel import gptq_matmul

torch.manual_seed(1)
dev='cuda'
bits=4; maxq=15; ipb=32//bits
M,N,K,gs=16,32,72,72  # K%16=8 tail; single group isolates tail masking
num_groups=1
q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
wp=torch.zeros((K+ipb-1)//ipb,N,device=dev,dtype=torch.int32)
for i in range(ipb):
    kk=slice(i*ipb,min((i+1)*ipb,K))
    wp[i] |= torch.cat([q[kk], torch.zeros((i*ipb+ipb-K)%ipb if i*ipb+ipb>K else 0,dtype=torch.int32,device=dev)]) << (i*bits)
scales=torch.rand(num_groups,N,device=dev,dtype=torch.float32)+0.5
zq=torch.randint(0,15,(num_groups,N),device=dev,dtype=torch.int32)
zeros=torch.zeros(num_groups,N//ipb,device=dev,dtype=torch.int32)
for i in range(ipb):
    zeros |= zq[:, i*ipb:(i+1)*ipb] << (i*bits)
a=torch.randn(M,K,device=dev,dtype=torch.float32)
zero_unp=zq.float()+1
deq=(q.float()-zero_unp)*scales
ref=a@deq
c=gptq_matmul(a,wp,scales,zeros,gs,bits)
err=(c-ref).abs().max().item()
rel=err/ref.abs().max().item()
print(json.dumps({"max_abs_err":err,"max_ref_abs":ref.abs().max().item(),"rel_err":rel,"K":K,"Kmod16":K%16}))