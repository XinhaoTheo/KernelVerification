
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_31/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev='cuda'
def ref(q,k,v,cuq,cuk,mask,qbs,kbs,causal,scale,hd,nhq,nhk):
    out=torch.empty_like(q)
    for b in range(len(cuq)-1):
        qs,qe=int(cuq[b]),int(cuq[b+1]); ks,ke=int(cuk[b]),int(cuk[b+1])
        for i in range((qe-qs+qbs-1)//qbs):
            mi_s,mi_e=qs+i*qbs,min(qs+(i+1)*qbs,qe)
            for j in range((ke-ks+kbs-1)//kbs):
                if not mask[:, i, j].all(): continue
                mj_s,mj_e=ks+j*kbs,min(ks+(j+1)*kbs,ke)
                if causal and (mi_s-qs) + (ke-ks)-(qe-qs) >= mj_s-ks + kbs: pass
                qb=q[mi_s:mi_e]; kb=k[mj_s:mj_e]; vb=v[mj_s:mj_e]
                s=(qb@kb.transpose(-1,-2))*scale
                if causal:
                    # causal within sequence: q pos -> key pos allowed if key<=q+offset
                    off=(ke-ks)-(qe-qs)
                    qi=torch.arange(mi_s-qs,mi_e-qs,device=q.device)[:,None]
                    kj=torch.arange(mj_s-ks,mj_e-ks,device=q.device)[None,:]
                    s=s.masked_fill(kj > qi+off, float('-inf'))
                p=torch.softmax(s,dim=-1)
                out[mi_s:mi_e]=(p@vb)
    return out

results={}
for lens_q, lens_k, kbs in [([5,3],[5,3],4), ([6,2],[6,2],4), ([9,7,4],[9,7,4],8)]:
    cuq=torch.tensor([0]+list(torch.tensor(lens_q).cumsum(0)),device=dev,dtype=torch.int32)
    cuk=torch.tensor([0]+list(torch.tensor(lens_k).cumsum(0)),device=dev,dtype=torch.int32)
    tq,tk=int(cuq[-1]),int(cuk[-1])
    nhq,nhk,hd=2,2,16
    q=torch.randn(tq,nhq,hd,device=dev); k=torch.randn(tk,nhk,hd,device=dev); v=torch.randn(tk,nhk,hd,device=dev)
    nqb=(tq+kbs-1)//kbs  # blocks per batch ceil(l/qbs)
    total_q_blocks=sum((l+kbs-1)//kbs for l in lens_q)
    total_k_blocks=sum((l+kbs-1)//kbs for l in lens_k)
    mask=torch.ones(nhq,total_q_blocks,total_k_blocks,dtype=torch.bool,device=dev)
    scale=hd**-0.5
    got=m.block_sparse_attention(q,k,v,cuq,cuk,mask,kbs,kbs,causal=False,softmax_scale=scale)
    exp=ref(q,k,v,cuq,cuk,mask,kbs,kbs,False,scale,hd,nhq,nhk)
    err=(got-exp).abs().max().item()
    results[f"lens={lens_q},kbs={kbs}"]=err
print(json.dumps({"errors":results,"total_k_divisible":[ [sum([5,3])%4==0, sum([6,2])%4==0, sum([9,7,4])%8==0] ]}))
