import torch, json, sys
import triton
sys.path.insert(0,'/root/cases/case_33')
from kernel import gptq_matmul

torch.manual_seed(1)
dev='cuda'
bits=4; maxq=15; ipb=32//bits
M,N,K,gs=16,32,72,72  # K%16=8 tail, single group isolates tail masking
num_groups=1
q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
nwp=(K+ipb-1)//ipb
wp=torch.zeros(nwp,N,device=dev,dtype=torch.int32)
for i in range(ipb):
    k0=i*ipb
    if k0<K:
        seg=q[k0:min(k0+ipb,K)]
        wp[:seg.shape[0] if False else nwp] = wp[:nwp]  # no-op keep dtype
        # pack segment into word rows k0//ipb..: each word row packs 8 k's; word index = k0//ipb only if aligned
for i in range(ipb):
    k0=i*ipb
    if k0<K:
        seg=q[k0:min(k0+ipb,K)]
        # only word row k0//ipb if k0 aligned; here k0 aligned always (i*ipb)
        wp[k0//ipb] |= seg << (i*bits)
scales=torch.rand(num_groups,N,device=dev,dtype=torch.float32)+0.5
zq=torch.randint(0,15,(num_groups,N),device=dev,dtype=torch.int32)
zeros=torch.zeros(num_groups,N//ipb,device=dev,dtype=torch.int32)
for j in range(N//ipb):
    for i in range(ipb):
        zeros[:,j] |= zq[:, j*ipb+i] << (i*bits)
a=torch.randn(M,K,device=dev,dtype=torch.float32)
deq=(q.float()-(zq.float()+1))*scales
ref=a@deq
c=gptq_matmul(a,wp,scales,zeros,gs,bits)
err=(c-ref).abs().max().item()
rel=err/ref.abs().max().item()
print(json.dumps({"max_abs_err":err,"max_ref_abs":ref.abs().max().item(),"rel_err":rel,"K":K,"Kmod16":K%16,"group_size":gs}))