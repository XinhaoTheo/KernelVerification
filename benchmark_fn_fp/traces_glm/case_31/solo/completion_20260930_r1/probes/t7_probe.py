
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_31/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev='cuda'
BS=32  # >= 16, power of 2

def ref(q,k,v,cuq,cuk,mask,qbs,kbs,scale):
    # dense per-sequence reference, non-causal
    out=torch.zeros_like(q)
    for b in range(len(cuq)-1):
        qs,qe=int(cuq[b]),int(cuq[b+1]); ks,ke=int(cuk[b]),int(cuk[b+1])
        qb=q[qs:qe]; kb=k[ks:ke]; vb=v[ks:ke]
        s=(qb@kb.transpose(-1,-2))*scale
        p=torch.softmax(s,dim=-1)
        out[qs:qe]=p@vb
    return out

results={}
cases=[([40,24],[40,24]), ([50,14],[50,14]), ([33,31],[33,31])]
for lens_q, lens_k in cases:
    cuq=torch.tensor([0]+list(torch.tensor(lens_q).cumsum(0)),device=dev,dtype=torch.int32)
    cuk=torch.tensor([0]+list(torch.tensor(lens_k).cumsum(0)),device=dev,dtype=torch.int32)
    tq,tk=int(cuq[-1]),int(cuk[-1])
    hd=64; nhq,nhk=2,2
    q=torch.randn(tq,nhq,hd,device=dev); k=torch.randn(tk,nhk,hd,device=dev); v=torch.randn(tk,nhk,hd,device=dev)
    tqb=sum((l+BS-1)//BS for l in lens_q); tkb=sum((l+BS-1)//BS for l in lens_k)
    mask=torch.ones(nhq,tqb,tkb,dtype=torch.bool,device=dev)
    scale=hd**-0.5
    got=m.block_sparse_attention(q,k,v,cuq,cuk,mask,BS,BS,causal=False,softmax_scale=scale)
    exp=ref(q,k,v,cuq,cuk,mask,BS,BS,scale)
    d=(got-exp).abs()
    # per-row max error
    rowerr=d.max(dim=-1).values.max(dim=-1).values  # [tq]
    results[f"lens={lens_q}"]={
      "max_abs_err": d.max().item(),
      "total_div_by_32": (tk%BS==0),
      "per_seq_div": [ (l%BS==0) for l in lens_k ],
      "err_rows_seq0_end": rowerr[:lens_q[0]].max().item(),
      "err_rows_seq1": rowerr[lens_q[0]:].max().item(),
    }
print(json.dumps(results))
