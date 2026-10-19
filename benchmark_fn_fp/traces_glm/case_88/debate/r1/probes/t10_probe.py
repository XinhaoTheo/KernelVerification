import torch, sys, json
sys.path.insert(0, '/root/cases/case_161')
import kernel
dev='cuda'
def oracle(t,f):
    x=t.to(torch.int64); w=f.to(torch.int64)
    out=torch.zeros(x.shape[0],4,4,dtype=torch.int64,device=x.device)
    for i in range(3):
        for j in range(3):
            out+=x[:,i:i+4,j:j+4]*w[:,i,j,None,None]
    return out.to(torch.int32)
gen=torch.Generator(device='cpu').manual_seed(1)
mism=0; total=0
# strongly asymmetric filters: one-hot basis filters at each of 9 positions
for i in range(3):
    for j in range(3):
        f=torch.zeros(16,3,3,dtype=torch.int8); f[:,i,j]=127
        t=torch.randint(-128,128,(16,6,6),generator=gen,dtype=torch.int8)
        t=t.to(dev); f=f.to(dev)
        got=kernel.run(t,f); exp=oracle(t,f)
        m=(got!=exp).sum().item(); total+=got.numel(); mism+=m
        if m: print('asym mismatch at filter pos',(i,j),m)
# random asymmetric filters
t=torch.randint(-128,128,(512,6,6),generator=gen,dtype=torch.int8).to(dev)
f=torch.randint(-128,128,(512,3,3),generator=gen,dtype=torch.int8).to(dev)
got=kernel.run(t,f); exp=oracle(t,f)
mism+=(got!=exp).sum().item(); total+=got.numel()
# explicit example from claim
t=torch.randint(-128,128,(4,6,6),generator=gen,dtype=torch.int8).to(dev)
f=torch.tensor([[127,0,0],[0,0,0],[0,0,-127]],dtype=torch.int8,device=dev).repeat(4,1,1)
got=kernel.run(t,f); exp=oracle(t,f)
mism+=(got!=exp).sum().item(); total+=got.numel()
print(json.dumps({'claim':'c2','total_positions':total,'mismatches':mism}))