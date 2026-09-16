import torch, json, sys
sys.path.insert(0, '/root/cases/case_33')
from kernel import gptq_matmul

dev='cuda'
bits=4; ipb=8
sh=torch.arange(ipb,device=dev,dtype=torch.int64)*bits

def pack_case(K,N,gs,seed):
    torch.manual_seed(seed)
    ngroups=(K+gs-1)//gs
    q=torch.randint(0,16,(K,N),device=dev,dtype=torch.int64)
    nwp=K//ipb  # K%8==0 in all cases used
    qv=q.reshape(nwp,ipb,N)
    wp=((qv<<sh[None,:,None])).bitwise_or(0) if False else None
    wp=torch.zeros(nwp,N,device=dev,dtype=torch.int64)
    for i in range(ipb):
        wp |= qv[:,i,:] << (i*bits)
    wp32=wp.to(torch.int32)
    scales=torch.rand(ngroups,N,device=dev,dtype=torch.float32)+0.5
    zq=torch.randint(0,15,(ngroups,N),device=dev,dtype=torch.int64)
    zeros=torch.zeros(ngroups,N//ipb,device=dev,dtype=torch.int64)
    zqg=zq.reshape(ngroups,N//ipb,ipb)
    for i in range(ipb):
        zeros |= zqg[:,:,i] << (i*bits)
    zeros32=zeros.to(torch.int32)
    a=torch.randn(16,K,device=dev,dtype=torch.float32)
    return a,wp32,scales,zeros32,zq,q

def ref_with(a,q,zq,scales,g):
    deq=(q.float()-(zq.float()[g]+1))*scales[g]
    return a@deq

results={}
for (K,N,gs,tag) in [(64,32,32,'control'),(64,32,48,'c1_case')]:
    a,wp,scales,zeros,zq,q=pack_case(K,N,gs,seed=0)
    g=torch.arange(K,device=dev,dtype=torch.int64)//gs
    ref=ref_with(a,q,zq,scales,g)
    c=gptq_matmul(a,wp,scales,zeros,gs,4)
    err=(c-ref).abs().max().item()
    ngc=(K+gs-1)//gs
    gf=torch.clamp(g,max=ngc-2)
    refc=ref_with(a,q,zq,scales,gf)
    errc=(c-refc).abs().max().item()
    results[tag]={"max_err_vs_contract_ref":err,"max_err_vs_clamped_ref":errc,"ref_scale":ref.abs().max().item(),"K":K,"gs":gs,"ngroups_contract":ngc}
print(json.dumps(results))
