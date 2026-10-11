import sys, json, torch
sys.path.insert(0, "/root/cases/case_121")
import kernel

def gen(M,K,N,gs,seed):
    dev="cuda"
    rng=torch.Generator(device="cpu").manual_seed(seed)
    groups=(K+gs-1)//gs
    x=(torch.rand((M,K),generator=rng)-.5).half()
    codes=torch.randint(0,16,(((K+7)//8)*8,N),generator=rng,dtype=torch.int64)
    qw=torch.zeros(((K+7)//8,N),dtype=torch.int64)
    for i in range(8): qw|=codes[i::8]<<(4*i)
    zc=torch.randint(0,16,(groups,((N+7)//8)*8),generator=rng,dtype=torch.int64)
    qz=torch.zeros((groups,(N+7)//8),dtype=torch.int64)
    for i in range(8): qz|=zc[:,i::8]<<(4*i)
    sc=(.002+.029*torch.rand((groups,N),generator=rng)).half()
    ids=(torch.arange(K)//gs)[torch.randperm(K,generator=rng)].int()
    return (x.to(dev),qw.int().to(dev),sc.to(dev),qz.int().to(dev),ids.to(dev),gs)

results=[]
for (M,K,N,gs) in [(1,1,1,16),(1,5,8,16),(2,8,16,16),(3,13,17,16),(4,15,32,16),(7,31,65,16),(65,16,65,16),(65,13,17,16),(7,17,32,32),(7,65,96,32)]:
    try:
        t=gen(M,K,N,gs,seed=K*77+N)
        out=kernel.run(*t); ref=kernel.reference(*t)
        err=(out-ref).abs(); tol=0.01+0.001*ref.abs()
        results.append({"case":[M,K,N,gs],"max_abs_err":err.max().item(),"n_violations":int((err>tol).sum())})
    except Exception as e:
        results.append({"case":[M,K,N,gs],"error":str(e)[:300]})
print(json.dumps({"claim":"c3","metric":"max_abs_err + violation count vs reference for small K / edge M","results":results}))
