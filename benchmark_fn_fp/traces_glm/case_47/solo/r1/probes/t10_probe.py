import torch, json, sys, random
sys.path.insert(0, "/root/cases/case_120")
import kernel as K
dev="cuda"
def gen(m,kk,n,gs,seed):
    rng=torch.Generator(device="cpu").manual_seed(seed)
    G=(kk+gs-1)//gs
    x=(torch.rand((m,kk),generator=rng)-.5).half().to(dev)
    codes=torch.randint(0,16,(((kk+7)//8)*8,n),generator=rng,dtype=torch.int64)
    qw=torch.zeros(((kk+7)//8,n),dtype=torch.int64)
    for i in range(8): qw|=codes[i::8]<<(4*i)
    qw=qw|(torch.randint(0,2,qw.shape,generator=rng,dtype=torch.int64)*(1<<31))
    zc=torch.randint(0,16,(G,((n+7)//8)*8),generator=rng,dtype=torch.int64)
    qz=torch.zeros((G,(n+7)//8),dtype=torch.int64)
    for i in range(8): qz|=zc[:,i::8]<<(4*i)
    qz=qz|(torch.randint(0,2,qz.shape,generator=rng,dtype=torch.int64)*(1<<31))
    sc=(.002+.029*torch.rand((G,n),generator=rng)).half().to(dev)
    ids=(torch.arange(kk)//gs)[torch.randperm(kk,generator=rng)].int().to(dev)
    return (x,qw.int().to(dev),sc,qz.int().to(dev),ids,gs)
fails=[];tested=0;maxexcess=-1e9
random.seed(1)
for trial in range(120):
    gs=random.choice([16,32,64]); kk=random.randint(1,128)
    if gs>16 and kk<gs: kk=gs
    m=random.randint(1,65); n=random.randint(1,96); seed=trial
    x,qw,sc,qz,ids,gs_=gen(m,kk,n,gs,seed)
    out=K.run(x,qw,sc,qz,ids,gs_); ref=K.reference(x,qw,sc,qz,ids,gs_)
    err=(out.double()-ref.double()).abs(); tol=0.01+0.001*ref.double().abs()
    tested+=1; maxexcess=max(maxexcess,float((err-tol).max()))
    if bool((err>tol).any()):
        fails.append(dict(m=m,k=kk,n=n,gs=gs,seed=seed,maxerr=float(err.max()),bad=int((err>tol).sum())))
        if len(fails)>=5: break
print(json.dumps({"tested":tested,"max_excess_over_tol":maxexcess,"fails":fails}),flush=True)
