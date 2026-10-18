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
gen=torch.Generator(device='cpu').manual_seed(2)
mism=0; total=0; ok_batches=[]
for B in [1,2,3,15,16,17,37,255,256,1000,2047,2048]:
    t=torch.randint(-128,128,(B,6,6),generator=gen,dtype=torch.int8)
    f=torch.randint(-128,128,(B,3,3),generator=gen,dtype=torch.int8)
    t=t.to(dev); f=f.to(dev)
    got=kernel.run(t,f); exp=oracle(t,f)
    m=(got!=exp).sum().item(); total+=got.numel(); mism+=m
    if m: print('batch',B,'mismatch',m)
    else: ok_batches.append(B)
    del got, exp, t, f; torch.cuda.empty_cache()
print(json.dumps({'claim':'c3','batches_tested':[1,2,3,15,16,17,37,255,256,1000,2047,2048],'clean_batches':ok_batches,'total_positions':total,'mismatches':mism}))