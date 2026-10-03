import sys, torch, json, math
sys.path.insert(0, "/root/cases/case_115")
import kernel
dev="cuda"

def gen(b,qt,d,hk,group,page,lengths,window,seed):
    g=torch.Generator().manual_seed(seed)
    hq=hk*group
    pages=random.Random(seed).randint( max((max(lengths)+page-1)//page,1), 256)
    c = 512//page  # max allowed, >= ceil(max len / page) since len<=512
    q=(2*torch.rand((b,qt,hq,d),generator=g)-1).half()
    k=(2*torch.rand((pages,page,hk,d),generator=g)-1).half()
    v=(2*torch.rand(k.shape,generator=g)-1).half()
    tbl=torch.randint(0,pages,(b,c),generator=g).to(torch.int32)
    ln=torch.tensor(lengths,dtype=torch.int32)
    return tuple(t.to(dev) for t in (q,k,v,tbl,ln))+(window,)

import random
configs=[
 (1,1,128,4,4,64,[512],0),
 (1,33,64,1,1,64,[512],0),
 (1,33,64,1,1,64,[512],256),
 (1,33,64,1,1,64,[512],1),
 (3,8,64,4,2,32,[64,128,512],0),
 (3,8,64,4,2,32,[64,128,512],63),
 (4,5,32,3,1,16,[500,33,64,129],0),
 (4,5,32,3,1,16,[500,33,64,129],128),
 (1,2,64,1,8,16,[512],0),
 (2,20,64,2,2,32,[512,512],0),
 (2,20,64,2,2,32,[512,512],100),
 (1,6,32,4,4,16,[512],256),
 (1,6,32,4,4,16,[512],255),
 (2,9,64,2,2,16,[97,113],41),
]
results=[]; worst=0.0; worst_cfg=None; nerr=0
for i,cfg in enumerate(configs):
    try:
        inputs=gen(*cfg, seed=2000+i)
        out=kernel.run(*inputs)
        ratio=kernel.error_ratio(out,inputs)
        fin=bool(torch.isfinite(out).all())
        results.append([i,cfg,round(ratio,4),fin])
        if ratio>worst: worst,worst_cfg=ratio,cfg
    except Exception as e:
        nerr+=1
        results.append([i,cfg,f"ERROR {type(e).__name__}: {e}",None])
print(json.dumps({"metric":"max contract error ratio over sweep (C<=512/S)","reason":"ratio<=1 is contract tolerance","results":results,"worst_ratio":worst,"worst_cfg":worst_cfg,"errors":nerr}))
