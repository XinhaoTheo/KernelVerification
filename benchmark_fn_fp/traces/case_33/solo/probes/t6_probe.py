
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_33/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
dev='cuda'

def build(M,K,N,gs,bits=4):
    per=32//bits; maxq=(1<<bits)-1
    ng=-(-K//gs)  # ceil per contract
    q=torch.randint(0,maxq+1,(K,N),device=dev,dtype=torch.int32)
    bp=torch.zeros((K//per,N),device=dev,dtype=torch.int32)
    for k in range(K):
        bp[k//per]|= (q[k]&maxq)<<((k%per)*bits)
    zq=torch.randint(0,maxq+1,(ng,N),device=dev,dtype=torch.int32)
    zp=torch.zeros((ng,N//per),device=dev,dtype=torch.int32)
    for n in range(N):
        zp[:,n//per]|=(zq[:,n]&maxq)<<((n%per)*bits)
    sc=(torch.rand((ng,N),device=dev)*0.1+0.01).float()
    a=torch.randn((M,K),device=dev)
    gidx=torch.arange(K,device=dev)//gs   # true grouping, ceil rows
    W=(q.float()-(zq[gidx].float()+1))*sc[gidx]   # kernel's zero+1 convention
    ref=a@W
    return a,bp,sc,zp,ref,gidx,q,zq,sc

out={}
for (M,K,N,gs) in [(32,64,32,32),(32,48,32,32)]:
    a,bp,sc,zp,ref,gidx,q,zq,scs=build(M,K,N,gs)
    c=m.gptq_matmul(a,bp,sc,zp,gs,bits=4)
    err=(c-ref).abs()
    # error attributable only to tail columns?
    ng_floor=K//gs
    tail=torch.arange(K,device=dev)>=ng_floor*gs
    key=f"M{M}_K{K}_N{N}_gs{gs}"
    out[key]={"scales_rows":sc.shape[0],"max_abs_err":err.max().item(),
              "ref_absmax":ref.abs().max().item(),
              "rel":(err.max()/ref.abs().max()).item(),
              "num_tail_cols":int(tail.sum().item())}
    if tail.any():
        # reference if tail cols wrongly used group ng_floor-1 (kernel's clamp)
        gbad=torch.clamp(torch.arange(K,device=dev)//gs,max=ng_floor-1)
        Wbad=(q.float()-(zq[gbad].float()+1))*scs[gbad]
        refbad=a@Wbad
        out[key]["max_abs_err_vs_clamped_ref"]=(c-refbad).abs().max().item()
print(json.dumps(out,indent=1))
