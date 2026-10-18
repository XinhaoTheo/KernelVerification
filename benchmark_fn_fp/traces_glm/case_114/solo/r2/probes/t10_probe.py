import torch, math, json, sys
torch.cuda.init()
sys.path.insert(0, '/root/cases/case_114')
from kernel import run, reference

dev='cuda'
results=[]
for seed,(b,qt,hq,hk,d,page,pages,lens,window) in enumerate([
    (1,4,2,1,64,16,16,[100],5),
    (1,9,2,2,64,16,16,[113],0),
    (2,33,8,2,32,32,32,[512,300],0),
    (1,1,16,4,128,64,64,[512],256),
    (2,17,4,4,64,32,32,[200,64],64),
]):
    g=torch.Generator().manual_seed(seed)
    q=(2*torch.rand((b,qt,hq,d),generator=g)-1).half()
    k=(2*torch.rand((pages,page,hk,d),generator=g)-1).half()
    v=(2*torch.rand((pages,page,hk,d),generator=g)-1).half()
    C=max((l+page-1)//page for l in lens)
    table=torch.zeros((b,C),dtype=torch.int32)
    for bi,l in enumerate(lens):
        perm=torch.randperm(pages,generator=g)[: (l+page-1)//page]
        table[bi,:len(perm)]=perm.to(torch.int32)
    lengths=torch.tensor(lens,dtype=torch.int32)
    qc,kc,vc,tc,lc=(t.to(dev) for t in (q,k,v,table,lengths))
    snaps=[t.clone() for t in (qc,kc,vc,tc,lc)]
    out=run(qc,kc,vc,tc,lc,window)
    tgt=reference(qc,kc,vc,tc,lc,window)
    err=(out.double()-tgt).abs(); tol=0.003+0.003*tgt.abs()
    unchanged=all(torch.equal(a,b_) for a,b_ in zip((qc,kc,vc,tc,lc),snaps))
    results.append({"case":seed,"max_viol":float((err/tol).max()),"finite":bool(torch.isfinite(out).all()),"unchanged":unchanged})
print(json.dumps({"cases":results,"all_ok":all(r["max_viol"]<=1.0 and r["finite"] and r["unchanged"] for r in results)}))