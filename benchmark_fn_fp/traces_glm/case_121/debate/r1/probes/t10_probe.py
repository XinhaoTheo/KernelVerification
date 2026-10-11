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
# force HIGH nibbles negative-signed: set bit31 in some words (those nibbles are unused for qweight along K, so
# instead make full words negative while keeping the used low nibbles: set bit31 -> word becomes negative int32,
# used nibbles (low 30 bits for 8 nibbles) unchanged.
qw=qw64.int()
qz=qz64.int()
qw[0,::3]=qw[0,::3]|(1<<31)
qz[:,0]=qz[:,0]|(1<<31)
qw=qw.to(dev); qz=qz.to(dev)
t=(x.to(dev),qw,sc.to(dev),qz,ids.to(dev),gs)
out=kernel.run(*t); ref=kernel.reference(*t)
err=(out-ref).abs(); tol=0.01+0.001*ref.abs()
print(json.dumps({"claim":"c2","M,K,N": [M,K,N],"n_negative_qw_words":int((qw<0).sum()),"n_negative_qz_words":int((qz<0).sum()),
 "max_abs_err":err.max().item(),"n_violations":int((err>tol).sum()),"tol_max":tol.max().item(),
 "metric":"max_abs_err vs reference under 0.01+0.001|target|, negative-word input"}))
