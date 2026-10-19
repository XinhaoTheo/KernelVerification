import torch, json, sys
import triton
sys.path.insert(0, '/root/cases/case_33')
from kernel import gptq_matmul

torch.manual_seed(0)
dev='cuda'
bits=4; maxq=15; ipb=32//bits  # 8 nibbles per int32
M,N,K,gs=16,32,64,48
num_groups=(K+gs-1)//gs  # 2 per contract (ceil)
q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
wp=torch.zeros(K//ipb,N,device=dev,dtype=torch.int32)
for i in range(ipb):
    wp |= q[i*ipb:(i+1)*ipb] << (i*bits)
scales=torch.rand(num_groups,N,device=dev,dtype=torch.float32)+0.5
zq=torch.randint(0,15,(num_groups,N),device=dev,dtype=torch.int32)
zeros=torch.zeros(num_groups,N//ipb,device=dev,dtype=torch.int32)
for j in range(N//ipb):
    for i in range(ipb):
        zeros[:,j] |= zq[:, j*ipb+i] << (i*bits)
a=torch.randn(M,K,device=dev,dtype=torch.float32)
# contract reference: g_idx = k//gs with ceil groups
g=torch.arange(K,device=dev)//gs
zero_unp=zq.float()[g]+1  # kernel adds 1 to zero point
deq=(q.float()-zero_unp)*scales[g]
ref=a@deq
c=gptq_matmul(a,wp,scales,zeros,gs,bits)
err=(c-ref).abs().max().item()
# control: reference using wrapper's clamped (floor) g_idx
gf=torch.clamp(torch.arange(K,device=dev)//gs, max=num_groups-2)
deqf=(q.float()-zq.float()[gf]-1+1)*0  # placeholder
deqf=(q.float()-(zq.float()[gf]+1))*scales[gf]
reff=a@deqf
errf=(c-reff).abs().max().item()
print(json.dumps({"max_abs_err_vs_contract":err,"max_abs_err_vs_clamped_ref":errf,"ref_scale":ref.abs().max().item(),"K":K,"group_size":gs,"num_groups_contract":num_groups,"Kmod16":K%16}))