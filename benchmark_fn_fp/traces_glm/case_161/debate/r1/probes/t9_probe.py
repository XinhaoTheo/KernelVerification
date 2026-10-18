import torch, sys
sys.path.insert(0, '/root/cases/case_161')
import kernel
dev='cuda'
def oracle(t,f):
    x=t.to(torch.int64); w=f.to(torch.int64)
    # independent 9-term sum via unfold
    out=torch.zeros(x.shape[0],4,4,dtype=torch.int64,device=x.device)
    for i in range(3):
        for j in range(3):
            out+=x[:,i:i+4,j:j+4]*w[:,i,j,None,None]
    return out.to(torch.int32)
mism=0; total=0
gen=torch.Generator(device='cpu')
cases=[]
# adversarial: all +127, all -128, mixed signs, corners
for vals in [(127,127),(-128,-128),(127,-128),(-128,127)]:
    t=torch.full((8,6,6),vals[0],dtype=torch.int8); f=torch.full((8,3,3),vals[1],dtype=torch.int8)
    cases.append((t,f))
# random extremes
t=torch.randint(-128,128,(512,6,6),generator=gen,dtype=torch.int8)
f=torch.randint(-128,128,(512,3,3),generator=gen,dtype=torch.int8)
cases.append((t,f))
# 2000 random batches of 16
t=torch.randint(-128,128,(2000,6,6),generator=gen,dtype=torch.int8)
f=torch.randint(-128,128,(2000,3,3),generator=gen,dtype=torch.int8)
cases.append((t,f))
for t,f in cases:
    t=t.to(dev); f=f.to(dev)
    got=kernel.run(t,f)
    exp=oracle(t,f)
    m=(got!=exp).sum().item(); total+=got.numel(); mism+=m
    if m:
        idx=(got!=exp).nonzero()[0].tolist()
        print('MISMATCH sample', idx, got.flatten()[idx[0]].item(), exp.flatten()[idx[0]].item())
print({'total_positions':total,'mismatches':mism})
import json; print(json.dumps({'claim':'c1','total_positions':total,'mismatches':mism}))