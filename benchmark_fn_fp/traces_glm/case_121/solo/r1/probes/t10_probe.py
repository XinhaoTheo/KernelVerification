import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_121/kernel.py")
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
dev="cuda"

def indep_ref(x,qw,sc,qz,ids,gs):
    m,k = x.shape; n = qw.shape[1]
    cols = torch.arange(n, device=x.device)
    D = torch.empty((k,n),dtype=torch.float64, device=x.device)
    for r in range(k):
        q = (qw[r//8].long() >> (4*(r%8))) & 15
        g = int(ids[r].item())
        z = ((qz[g, cols//8].long() >> (4*(cols%8))) & 15) + 1
        D[r] = ((q - z).double() * sc[g].double()).half().double()
    return x.double() @ D

benign_fails=0; benign_tests=0; bw=0.0
for seed in range(3):
    for gs in (16,32,64):
        x,qw,sc,qz,ids,g = K.make_inputs(device=dev, rows=7, input_width=64, output_width=65, group_size=gs, seed=seed)
        out = K.run(x,qw,sc,qz,ids,g)
        ref = indep_ref(x,qw,sc,qz,ids,g)
        err=(out.double()-ref).abs(); tol=0.01+0.001*ref.abs()
        benign_tests+=1; bw=max(bw,(err/tol).max().item())
        if (err>tol).any(): benign_fails+=1

# tiny controlled benign case
rng=torch.Generator("cpu").manual_seed(1)
k,n,m,gs=16,8,2,16
codes=torch.randint(0,16,(k,n),generator=rng)
qw=torch.zeros(((k+7)//8,n),dtype=torch.int32)
for i in range(8): qw |= codes[i::8].int() << (4*i)
zc=torch.randint(0,16,(1,n),generator=rng)
qz=torch.zeros(1,(n+7)//8,dtype=torch.int32)
for i in range(8): qz |= zc[:,i::8].int() << (4*i)
sc=(.002+.029*torch.rand((1,n),generator=rng)).half()
x=(torch.rand((m,k),generator=rng)-.5).half()
ids=torch.zeros(k,dtype=torch.int32)
a=[t.to(dev) for t in (x,qw,sc,qz,ids)]
out=K.run(*a,gs); ref=indep_ref(*a,gs)
err=(out.double()-ref).abs(); tol=0.01+0.001*ref.abs()
tv=int((err>tol).sum()); tmax=(err/tol).max().item()
print(json.dumps({"benign_tests":benign_tests,"benign_fails":benign_fails,"benign_worst_ratio":bw,"tiny_viol":tv,"tiny_ratio":tmax,"tiny_maxerr":err.max().item()}))