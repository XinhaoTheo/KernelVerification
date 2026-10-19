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
    return (q,k,v,tbl,ln,window,g)

def poison(k,v,q,tbl,ln,window,mode):
    # mode 'n': NaN everywhere reachable-kv is kept; poison all slots NOT reachable by any allowed position
    k=k.clone(); v=v.clone()
    b,qt,hq,d=q.shape
    page,hk=k.shape[1],k.shape[2]
    for i in range(b):
        length=int(ln[i])
        lo = max(0, length-qt-window+1) if window else 0
        reachable=set(range(lo,length))
        used_pages=set(int(tbl[i,t//page]) for t in reachable)
        # poison slots not reachable: within used pages, positions not allowed; and all unused pages
        for p in range(k.shape[0]):
            if p not in used_pages:
                k[p]=float('nan'); v[p]=float('nan')
        for p in used_pages:
            for pos in range(page):
                tset=set()  # any logical position mapping to (p,pos)?
                # find logical positions in [0,length) mapping here
                for t in range(length):
                    if int(tbl[i,t//page])==p and t%page==pos and t in reachable:
                        tset.add(t)
                if not tset:
                    k[p,pos]=float('nan'); v[p,pos]=float('nan')
    # poison block_table entries beyond ceil(length/page)
    tbl=tbl.clone()
    for i in range(b):
        need=(int(ln[i])+page-1)//page
        tbl[i,need:]=99999
    return k,v,tbl

cases=[]
for seed,window in [(3,41),(4,0),(5,256),(6,1),(7,17)]:
    q,k,v,tbl,ln,window,g=build(seed,window)
    # reference on clean k,v first
    out_clean=kernel.run(q,k,v,tbl,ln,window)
    ref=kernel.reference(q,k,v,tbl,ln,window)
    ratio_clean=float(((out_clean.double()-ref).abs()/(0.003+0.003*ref.abs())).max())
    kp,vp,tblp=poison(k,v,q,tbl,ln,window)
    # validate inputs manually? validate would fail on NaN in cache -> use raw kernel? kernel.run validates. bypass by constructing where unreachable slots NaN but validate only checks reachable slots -> passes.
    out_p=kernel.run(q,kp,vp,tblp,ln,window)
    finite=bool(torch.isfinite(out_p).all())
    diff=float((out_p.float()-out_clean.float()).abs().max())
    ratio_p=float(((out_p.double()-ref).abs()/(0.003+0.003*ref.abs())).max())
    cases.append([seed,window,ratio_clean,ratio_p,diff,finite])
print(json.dumps({"metric":"contract ratio clean vs NaN-poisoned irrelevant slots","reason":"poisoned irrelevant slots must not change output; ratio<=1 and diff~0","cases":cases,"all_finite":all(c[5] for c in cases)}))
