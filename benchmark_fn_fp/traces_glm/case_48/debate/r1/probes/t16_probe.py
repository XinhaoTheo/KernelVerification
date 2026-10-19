import sys, json, torch
sys.path.insert(0, "/root/cases/case_121")
import kernel

dev="cuda"
M,K,N,gs=7,64,65,16
rng=torch.Generator(device="cpu").manual_seed(7)
groups=(K+gs-1)//gs
x=(torch.rand((M,K),generator=rng)-.5).half()
codes=torch.randint(0,16,(((K+7)//8)*8,N),generator=rng,dtype=torch.int64)
qw64=torch.zeros(((K+7)//8,N),dtype=torch.int64)
for i in range(8): qw64|=codes[i::8]<<(4*i)
zc=torch.randint(0,16,(groups,((N+7)//8)*8),generator=rng,dtype=torch.int64)
qz64=torch.zeros((groups,(N+7)//8),dtype=torch.int64)
for i in range(8): qz64|=zc[:,i::8]<<(4*i)
sc=(.002+.029*torch.rand((groups,N),generator=rng)).half()
ids=(torch.arange(K)//gs)[torch.randperm(K,generator=rng)].int()

def trial(qw_np, qz_np, tag):
    qw=qw_np.to(dev); qz=qz_np.to(dev)
    t=(x.to(dev),qw,sc.to(dev),qz,ids.to(dev),gs)
    out=kernel.run(*t); ref=kernel.reference(*t)
    err=(out-ref).abs(); tol=0.01+0.001*ref.abs()
    return {"tag":tag,"n_negative_qw":int((qw<0).sum()),"n_negative_qz":int((qz<0).sum()),
            "max_abs_err":err.max().item(),"n_violations":int((err>tol).sum())}

base_qw=qw64.int(); base_qz=qz64.int()
neg_qw=base_qw.clone(); neg_qw[0,::3]=neg_qw[0,::3]|(1<<31)
neg_qz=base_qz.clone(); neg_qz[:,0]=neg_qz[:,0]|(1<<31)
r=[trial(base_qw,base_qz,"control_no_negative"),
   trial(neg_qw,base_qz,"negative_qweight_only"),
   trial(base_qw,neg_qz,"negative_qzeros_only")]
print(json.dumps({"claim":"c2","M,K,N":[M,K,N],
 "metric":"paired comparison: violations with vs without negative words on identical low nibbles; isolates sign-extension effect from general packing bug",
 "results":r}))
