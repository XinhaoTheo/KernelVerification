import torch, sys, json
sys.path.insert(0, '/root/cases/case_160')
import kernel as K
dev='cuda'
# structured probes: constant tiles with unit impulse filters; also check divisor via a known linear case
# case 1: all-ones tile, impulse at each filter position -> output should equal a shifted ones patch
tile=torch.ones((1,6,6),dtype=torch.int8)
fails=[]
for i in range(3):
    for j in range(3):
        f=torch.zeros((1,3,3),dtype=torch.int8); f[0,i,j]=1
        out=K.run(tile.to(dev),f.to(dev))
        ref=K.reference(tile,f).to(dev)
        if not torch.equal(out,ref): fails.append((i,j,out.tolist(),ref.tolist()))
# case 2: identity-like scaling: tile all 7, filter single value 3 -> output constant 21
t7=torch.full((1,6,6),7,dtype=torch.int8)
f3=torch.zeros((1,3,3),dtype=torch.int8); f3[0,1,1]=3
out=K.run(t7.to(dev),f3.to(dev)); ref=K.reference(t7,f3).to(dev)
scale_ok=bool(torch.equal(out,ref))
# case 3: probe exact quotient: impulse filter value v on ones tile should scale linearly
lin_ok=True
for v in (-128,127,1,-1,37):
    f=torch.zeros((1,3,3),dtype=torch.int8); f[0,0,0]=v
    out=K.run(torch.ones((1,6,6),dtype=torch.int8).to(dev),f.to(dev))
    ref=K.reference(torch.ones((1,6,6),dtype=torch.int8),f).to(dev)
    if not torch.equal(out,ref): lin_ok=False
print(json.dumps({"impulse_failures":len(fails),"fails":fails[:3],
"ones_tile_3_filter_ok":scale_ok,"linear_v_ok":lin_ok}))
