import sys, torch, json
sys.path.insert(0, "/root/cases/case_115")
import kernel
dev="cuda"

def build(seed, window):
    g=torch.Generator().manual_seed(seed)
    b,qt,d,hk,group,page=2,9,64,2,2,16
    hq=4; lengths=[97,113]
    pages=20; c=8
    q=(2*torch.rand((b,qt,hq,d),generator=g)-1).half()
    k=(2*torch.rand((pages,page,hk,d),generator=g)-1).half()
    v=(2*torch.rand(k.shape,generator=g)-1).half()
    tbl=torch.randint(0,pages,(b,c),generator=g).to(torch.int32)
    ln=torch.tensor(lengths,dtype=torch.int32)
    return tuple(t.to(dev) for t in (q,k,v,tbl,ln))+(window,)

def poison(k,v,q,tbl,ln,window):
    k=k.clone(); v=v.clone()
    b,qt,hq,d=q.shape
    page,hk=k.shape[1],k.shape[2]
    for i in range(b):
        length=int(ln[i].item())
        lo = max(0, length-qt-window+1) if window else 0
        reachable=set(range(lo,length))
        used_pages=set(int(tbl[i,t//page].item()) for t in reachable)
        for p in range(k.shape[0]):
            if p not in used_pages:
                k[p]=float('nan'); v[p]=float('nan')
        for p in used_pages:
            for pos in range(page):
                ok=any(int(tbl[i,t//page].item())==p and t%page==pos and t in reachable for t in range(length))
                if not ok:
                    k[p,pos]=float('nan'); v[p,pos]=float('nan')
    tbl=tbl.clone()
    for i in range(b):
        need=(int(ln[i].item())+page-1)//page
        tbl[i,need:]=999999
    return k,v,tbl

cases=[]
for seed,window in [(3,41),(4,0),(5,256),(6,1),(7,17)]:
    q,k,v,tbl,ln,window=build(seed,window)
    out_clean=kernel.run(q,k,v,tbl,ln,window)
    ref=kernel.reference(q,k,v,tbl,ln,window)
    ratio_clean=float(((out_clean.double()-ref).abs()/(0.003+0.003*ref.abs())).max())
    kp,vp,tblp=poison(k,v,q,tbl,ln,window)
    out_p=kernel.run(q,kp,vp,tblp,ln,window)
    finite=bool(torch.isfinite(out_p).all())
    diff=float((out_p.float()-out_clean.float()).abs().max())
    ratio_p=float(((out_p.double()-ref).abs()/(0.003+0.003*ref.abs())).max())
    cases.append([seed,window,round(ratio_clean,4),round(ratio_p,4),diff,finite])
print(json.dumps({"metric":"contract ratio clean vs NaN-poisoned irrelevant slots (incl. dirty block_table entries beyond ceil(length/page))","reason":"irrelevant-slot NaN must not change output; require finite outputs and ratio<=1","cases":cases,"all_finite":all(c[5] for c in cases)}))
