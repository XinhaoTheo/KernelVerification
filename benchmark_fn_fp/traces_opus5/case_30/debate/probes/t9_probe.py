
import importlib.util, json, torch

spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_31/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

dev="cuda"; torch.manual_seed(1)

def blocks(cu,bs):
    cun=[0]
    for b in range(len(cu)-1):
        cun.append(cun[-1] + -(-(cu[b+1]-cu[b])//bs))
    return cun

def ref_attn(q,k,v,cu_q,cu_k,mask,qbs,kbs,scale,hsel):
    out=torch.zeros_like(q); B=len(cu_q)-1
    cu_nq=blocks(cu_q,qbs); cu_nk=blocks(cu_k,kbs)
    Hq=q.shape[1]; ratio=Hq//k.shape[1]
    for b in range(B):
        qs,qe=cu_q[b],cu_q[b+1]; ks,ke=cu_k[b],cu_k[b+1]
        Lq,Lk=qe-qs,ke-ks
        nqb=-(-Lq//qbs); nkb=-(-Lk//kbs)
        for hq in range(Hq):
            hm=hsel(hq); hk=hq//ratio
            allowed=torch.zeros(Lq,Lk,dtype=torch.bool,device=dev)
            for i in range(nqb):
                r0,r1=i*qbs,min((i+1)*qbs,Lq)
                for j in range(nkb):
                    c0,c1=j*kbs,min((j+1)*kbs,Lk)
                    if bool(mask[hm,cu_nq[b]+i,cu_nk[b]+j]): allowed[r0:r1,c0:c1]=True
            s=q[qs:qe,hq].float() @ k[ks:ke,hk].float().T * scale
            s=s.masked_fill(~allowed,float('-inf'))
            p=torch.nan_to_num(torch.softmax(s,-1),nan=0.0)
            out[qs:qe,hq]=(p@v[ks:ke,hk].float()).to(out.dtype)
    return out

D=16; Hq=4; Hk=2; ratio=2
cu_q=[0,8,16]; cu_k=[0,8,16]; qbs=kbs=4
S=16
q=torch.randn(S,Hq,D,device=dev); k=torch.randn(S,Hk,D,device=dev); v=torch.randn(S,Hk,D,device=dev)
cu_q_t=torch.tensor(cu_q,device=dev,dtype=torch.int32); cu_k_t=torch.tensor(cu_k,device=dev,dtype=torch.int32)
nqb=blocks(cu_q,qbs)[-1]; nkb=blocks(cu_k,kbs)[-1]
scale=D**-0.5

# per-query-head mask, rows distinct; diagonal always selected so no all-masked rows
mask=torch.zeros(Hq,nqb,nkb,device=dev,dtype=torch.bool)
for h in range(Hq):
    for i in range(nqb):
        mask[h,i,i]=True
mask[0,0,1]=True; mask[1,1,0]=True; mask[2,0,1]=False; mask[2,1,0]=True; mask[3,0,1]=True; mask[3,1,0]=False
mask[2,2,3]=True; mask[3,3,2]=True

out=m.block_sparse_attention(q,k,v,cu_q_t,cu_k_t,mask,qbs,kbs,False,None)
ref_qhead=ref_attn(q,k,v,cu_q,cu_k,mask,qbs,kbs,scale,lambda h:h)
ref_kvhead=ref_attn(q,k,v,cu_q,cu_k,mask,qbs,kbs,scale,lambda h:h//ratio)

# mutate rows >= nhead_k and re-run
mask_mut=mask.clone(); mask_mut[2]=~mask_mut[2]; mask_mut[3]=~mask_mut[3]
for i in range(nqb):
    mask_mut[2,i,i]=True; mask_mut[3,i,i]=True
out_mut=m.block_sparse_attention(q,k,v,cu_q_t,cu_k_t,mask_mut,qbs,kbs,False,None)

print(json.dumps({
 "max_abs_err_vs_query_head_interp": (out-ref_qhead).abs().max().item(),
 "max_abs_err_vs_kv_head_interp": (out-ref_kvhead).abs().max().item(),
 "per_head_err_vs_qhead": [ (out[:,h]-ref_qhead[:,h]).abs().max().item() for h in range(Hq)],
 "per_head_err_vs_kvhead": [ (out[:,h]-ref_kvhead[:,h]).abs().max().item() for h in range(Hq)],
 "max_change_when_mask_rows_2_3_mutated": (out-out_mut).abs().max().item(),
 "nhead_q": Hq, "nhead_k": Hk, "block_mask_shape": list(mask.shape),
 "metric": "max abs error vs two candidate mask-head interpretations + sensitivity to mask rows >= nhead_k"
}))
