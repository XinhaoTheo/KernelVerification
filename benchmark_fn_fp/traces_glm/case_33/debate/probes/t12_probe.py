import torch, json, sys
sys.path.insert(0, '/root/cases/case_33')
from kernel import gptq_matmul

dev='cuda'
bits=4; ipb=8

def pack_case(K,N,gs,seed):
    torch.manual_seed(seed)
    ngroups_contract=(K+gs-1)//gs
    q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int32)
    nwp=K//ipb
    wp=torch.zeros(nwp,N,device=dev,dtype=torch.int32)
    for i in range(ipb):
        wp |= q[i*ipb:(i+1)*ipb] << (i*bits)
    scales=torch.rand(ngroups_contract,N,device=dev,dtype=torch.float32)+0.5
    zq=torch.randint(0,15,(ngroups_contract,N),device=dev,dtype=torch.int32)
    zeros=torch.zeros(ngroups_contract,N//ipb,device=dev,dtype=torch.int32)
    for j in range(N//ipb):
        for i in range(ipb):
            zeros[:,j] |= zq[:, j*ipb+i] << (i*bits)
    a=torch.randn(16,K,device=dev,dtype=torch.float32)
    return a,wp,scales,zeros,zq,q

def ref_with(a,q,zq,scales,g):
    deq=(q.float()-(zq.float()[g]+1))*scales[g]
    return a@deq, deq

results={}
# control: K divisible by gs and by 16
for (K,N,gs,tag) in [(64,32,32,'control'),(64,32,48,'c1_case')]:
    a,wp,scales,zeros,zq,q=pack_case(K,N,gs,seed=0)
    g=torch.arange(K,device=dev)//gs
    ref,deq=ref_with(a,q,zq,scales,g)
    c=gptq_matmul(a,wp,scales,zeros,gs,4)
    err=(c-ref).abs().max().item()
    # wrapper clamped reference (what kernel SHOULD produce if bug is only group clamp)
    ngc=(K+gs-1)//gs
    gf=torch.clamp(g,max=ngc-2)
    refc,_=ref_with(a,q,zq,scales,gf)
    errc=(c-refc).abs().max().item()
    results[tag]={"max_err_vs_contract_ref":err,"max_err_vs_clamped_ref":errc,"ref_scale":ref.abs().max().item(),"K":K,"gs":gs,"ngroups_contract":ngc}
print(json.dumps(results))
