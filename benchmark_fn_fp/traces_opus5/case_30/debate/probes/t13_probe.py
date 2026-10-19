
import importlib.util, json, torch

spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_31/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

dev="cuda"; torch.manual_seed(1)

def cu_nb(cu,bs):
    o=[0]
    for b in range(len(cu)-1):
        o.append(o[-1] + -(-(cu[b+1]-cu[b])//bs))
    return o

def ref_attn(q,k,v,cu_q,cu_k,mask,qbs,kbs,scale,hsel):
    out=torch.zeros_like(q)
    cnq=cu_nb(cu_q,qbs); cnk=cu_nb(cu_k,kbs)
    Hq=q.shape[1]; ratio=Hq//k.shape[1]
    for b in range(len(cu_q)-1):
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
                    if bool(mask[hm,cnq[b]+i,cnk[b]+j]): allowed[r0:r1,c0:c1]=True
            s=q[qs:qe,hq].float() @ k[ks:ke,hk].float().T * scale
            s=s.masked_fill(~allowed,float('-inf'))
            p=torch.nan_to_num(torch.softmax(s,-1),nan=0.0)
            out[qs:qe,hq]=(p@v[ks:ke,hk].float()).to(out.dtype)
    return out

D=16; Hq=4; Hk=2; ratio=Hq//Hk
qbs=kbs=16; cu=[0,32,64]; S=64
q=torch.randn(S,Hq,D,device=dev); k=torch.randn(S,Hk,D,device=dev); v=torch.randn(S,Hk,D,device=dev)
cut=torch.tensor(cu,device=dev,dtype=torch.int32)
nqb=cu_nb(cu,qbs)[-1]; nkb=cu_nb(cu,kbs)[-1]  # 4 and 4
scale=D**-0.5

mask=torch.zeros(Hq,nqb,nkb,device=dev,dtype=torch.bool)
for h in range(Hq):
    for i in range(nqb): mask[h,i,i]=True
mask[0,0,1]=True
mask[1,1,0]=True
mask[2,2,3]=True
mask[3,3,2]=True

out=m.block_sparse_attention(q,k,v,cut,cut,mask,qbs,kbs,False,None)
ref_q=ref_attn(q,k,v,cu,cu,mask,qbs,kbs,scale,lambda h:h)
ref_kv=ref_attn(q,k,v,cu,cu,mask,qbs,kbs,scale,lambda h:h//ratio)

mask_mut=mask.clone()
mask_mut[2]=~mask_mut[2]; mask_mut[3]=~mask_mut[3]
for i in range(nqb):
    mask_mut[2,i,i]=True; mask_mut[3,i,i]=True
out_mut=m.block_sparse_attention(q,k,v,cut,cut,mask_mut,qbs,kbs,False,None)

mask_mut01=mask.clone()
mask_mut01[0,0,1]=False; mask_mut01[1,1,0]=False
out_mut01=m.block_sparse_attention(q,k,v,cut,cut,mask_mut01,qbs,kbs,False,None)

print(json.dumps({
 "nhead_q":Hq,"nhead_k":Hk,"block_mask_shape":list(mask.shape),
 "max_abs_err_vs_query_head_interp":(out-ref_q).abs().max().item(),
 "max_abs_err_vs_kv_head_interp":(out-ref_kv).abs().max().item(),
 "per_head_err_vs_qhead":[(out[:,h]-ref_q[:,h]).abs().max().item() for h in range(Hq)],
 "per_head_err_vs_kvhead":[(out[:,h]-ref_kv[:,h]).abs().max().item() for h in range(Hq)],
 "max_change_when_mask_rows_2_3_mutated":(out-out_mut).abs().max().item(),
 "max_change_when_mask_rows_0_1_mutated":(out-out_mut01).abs().max().item(),
 "metric":"max abs error vs two candidate mask-head interpretations plus sensitivity of output to mask rows >= nhead_k"
}))
