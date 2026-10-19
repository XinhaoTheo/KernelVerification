import torch, json, sys
sys.path.insert(0, '/root/cases/case_33')
from kernel import gptq_matmul

dev='cuda'
bits=4; ipb=8
M,N,K,gs=16,32,72,72   # single group, K%16=8 -> isolates tail k-block masking
torch.manual_seed(1)
num_groups=(K+gs-1)//gs  # 1
q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
nwp=(K+ipb-1)//ipb  # 9 words; last word has 8 valid k's? K=72 -> 9 full words, K%8=0
wp=torch.zeros(nwp,N,device=dev,dtype=torch.int32)
for i in range(ipb):
    k0=i*ipb
    seg=q[k0:min(k0+ipb,K)]
    wp[k0//ipb:(k0//ipb)+seg.shape[0]] |= seg << (i*bits)
scales=torch.rand(num_groups,N,device=dev,dtype=torch.float32)+0.5
zq=torch.randint(0,15,(num_groups,N),device=dev,dtype=torch.int32)
zeros=torch.zeros(num_groups,N//ipb,device=dev,dtype=torch.int32)
for j in range(N//ipb):
    for i in range(ipb):
        zeros[:,j] |= zq[:, j*ipb+i] << (i*bits)
a=torch.randn(M,K,device=dev,dtype=torch.float32)
deq=(q.float()-(zq.float()+1))*scales
ref=a@deq
c=gptq_matmul(a,wp,scales,zeros,gs,4)
err=(c-ref).abs().max().item()
print(json.dumps({"max_abs_err":err,"max_ref_abs":ref.abs().max().item(),"K":K,"Kmod16":K%16,"gs":gs,"num_groups":num_groups}))
