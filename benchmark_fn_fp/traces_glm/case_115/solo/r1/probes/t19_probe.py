import sys, torch
sys.path.insert(0, "/root/cases/case_115")
import kernel
dev="cuda"
g=torch.Generator().manual_seed(3)
b,qt,d,hk,page=2,9,64,2,16
hq=4; lengths=[97,113]; pages=20; c=8; window=41
q=(2*torch.rand((b,qt,hq,d),generator=g)-1).half()
k=(2*torch.rand((pages,page,hk,d),generator=g)-1).half()
v=(2*torch.rand(k.shape,generator=g)-1).half()
tbl=torch.randint(0,pages,(b,c),generator=g).to(torch.int32)
ln=torch.tensor(lengths,dtype=torch.int32)
q,k,v,tbl,ln=[t.to(dev) for t in (q,k,v,tbl,ln)]

# replicate poison logic on CPU copy and find the disagreement
for i in range(b):
    length=int(ln[i].item())
    lo = max(0, length-qt-window+1)
    reachable=set(range(lo,length))
    used=set(int(tbl[i,t//page].item()) for t in reachable)
    print("batch",i,"length",length,"lo",lo,"used",sorted(used))
    # validate reachable positions finite?
    for t in reachable:
        p=int(tbl[i,t//page].item()); pos=t%page
        if not torch.isfinite(k[p,pos]).all():
            print("LIVE-BUT-WOULD-POISON t=",t,"p",p,"pos",pos)
            # check poison predicate
            ok=any(int(tbl[i,tt//page].item())==p and tt%page==pos and tt in reachable for tt in range(length))
            print("poison predicate ok (True means keep):", ok, "tt candidates:", [tt for tt in range(length) if int(tbl[i,tt//page].item())==p and tt%page==pos])
