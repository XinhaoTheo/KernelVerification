import torch, json
import triton
import sys
sys.path.insert(0, '/root/cases/case_33')
from kernel import gptq_matmul

torch.manual_seed(0)
dev='cuda'
bits=4; maxq=15; ipb=32//bits
M,N,K,gs=16,32,64,48
num_groups=(K+gs-1)//gs  # 2 per contract
# pack b: unpacked q[k,n] in [0,15], pack 8 k's per int32 word
q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
wp=torch.zeros(K//ipb,N,device=dev,dtype=torch.int32)
for i in range(ipb):
    wp |= q[i*ipb:(i+1)*ipb] << (i*bits)
scales=torch.rand(num_groups,N,device=dev,dtype=torch.float32)+0.5
# zeros packed per 8 columns of N (kernel indexes zeros by column nibble)
zq=torch.randint(0,15,(num_groups,N),device=dev,dtype=torch.int32)
zeros=torch.zeros(num_groups,N//ipb,device=dev,dtype=torch.int32)
for i in range(ipb):
    zeros |= zq[:, i*ipb:(i+1)*ipb] << (i*bits)
a=torch.randn(M,K,device=dev,dtype=torch.float32)
# reference per contract: g_idx = k//gs, ceil rows
g=torch.arange(K,device=dev)//gs
deq=(q.float()-zq.float()[g]-0+1)*0  # placeholder
zero_unp=zq.float()[g]+1  # kernel adds 1 to zero point
deq=(q.float()-zero_unp)*scales[g]
ref=a@deq
c=gptq_matmul(a,wp,scales,zeros,gs,bits)
err=(c-ref).abs().max().item()
# also check what g_idx the wrapper effectively uses (clamp to floor groups)
print(json.dumps({"max_abs_err":err,"c_rel":(c.abs().max().item()),"K":K,"group_size":gs,"num_groups_contract":num_groups}))