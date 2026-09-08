
import importlib.util, json, torch

spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_31/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

dev="cuda"; torch.manual_seed(0)

def cu_nb(cu,bs):
    o=[0]
    for b in range(len(cu)-1):
        o.append(o[-1] + -(-(cu[b+1]-cu[b])//bs))
    return o

def ref_attn(q,k,v,cu_q,cu_k,mask,qbs,kbs,scale):
    out=torch.zeros_like(q)
    cnq=cu_nb(cu_q,qbs); cnk=cu_nb(cu_k,kbs)
    Hq=q.shape[1]; ratio=Hq//k.shape[1]
    for b in range(len(cu_q)-1):
        qs,qe=cu_q[b],cu_q[b+1]; ks,ke=cu_k[b],cu_k[b+1]
        Lq,Lk=qe-qs,ke-ks
        nqb=-(-Lq//qbs); nkb=-(-Lk//kbs)
        for hq in range(Hq):
            hk=hq//ratio
            allowed=torch.zeros(Lq,Lk,dtype=torch.bool,device=dev)
            for i in range(nqb):
                r0,r1=i*qbs,min((i+1)*qbs,Lq)
                for j in range(nkb):
                    c0,c1=j*kbs,min((j+1)*kbs,Lk)
                    if bool(mask[hk,cnq[b]+i,cnk[b]+j]): allowed[r0:r1,c0:c1]=True
            s=q[qs:qe,hq].float() @ k[ks:ke,hk].float().T * scale
            s=s.masked_fill(~allowed,float('-inf'))
            p=torch.nan_to_num(torch.softmax(s,-1),nan=0.0)
            out[qs:qe,hq]=(p@v[ks:ke,hk].float()).to(out.dtype)
    return out

D=16; H=1; qbs=kbs=8
cu=[0,3,8,16]; S=16
q=torch.randn(S,H,D,device=dev); k=torch.randn(S,H,D,device=dev); v=torch.randn(S,H,D,device=dev)
cut=torch.tensor(cu,device=dev,dtype=torch.int32)
nqb=cu_nb(cu,qbs)[-1]; nkb=cu_nb(cu,kbs)[-1]
mask=torch.ones(H,nqb,nkb,device=dev,dtype=torch.bool)
scale=D**-0.5

flag_k=bool(((cut[-1]-cut[0])%kbs==0).item())
flag_q=bool(torch.all((cut[1:]-cut[:-1])%qbs==0).item())

out=m.block_sparse_attention(q,k,v,cut,cut,mask,qbs,kbs,False,None)
ref=ref_attn(q,k,v,cu,cu,mask,qbs,kbs,scale)
err=(out-ref).abs()

def leak(qr0,qr1,kr0,kr1):
    s=q[qr0:qr1,0].float() @ k[kr0:kr1,0].float().T * scale
    return torch.softmax(s,-1) @ v[kr0:kr1,0].float()

# seq0 queries rows 0..2, its single k block spans rows 0..7 unmasked -> leaks rows 3..7
leak0=leak(0,3,0,8)
# seq1 queries rows 3..7, its single k block starts at 3, spans rows 3..10 -> leaks rows 8..10
leak1=leak(3,8,3,11)

e0_leak=(out[0:3,0]-leak0).abs().max().item()
e1_leak=(out[3:8,0]-leak1).abs().max().item()

# control: all sequence lengths multiples of 8
cu2=[0,8,16]; cut2=torch.tensor(cu2,device=dev,dtype=torch.int32)
nqb2=cu_nb(cu2,qbs)[-1]; nkb2=cu_nb(cu2,kbs)[-1]
mask2=torch.ones(H,nqb2,nkb2,device=dev,dtype=torch.bool)
out2=m.block_sparse_attention(q,k,v,cut2,cut2,mask2,qbs,kbs,False,None)
ref2=ref_attn(q,k,v,cu2,cu2,mask2,qbs,kbs,scale)

print(json.dumps({
 "cu_seqlens":cu,"k_block_size":kbs,"q_block_size":qbs,
 "EVEN_SEQ_KBLOCK_flag":flag_k,"EVEN_SEQ_QBLOCK_flag":flag_q,
 "max_abs_err_vs_same_seq_ref":err.max().item(),
 "err_seq0_rows_0_2":err[0:3].max().item(),
 "err_seq1_rows_3_7":err[3:8].max().item(),
 "err_seq2_rows_8_15":err[8:16].max().item(),
 "seq0_err_vs_leaked_ref":e0_leak,
 "seq1_err_vs_leaked_ref":e1_leak,
 "control_even_lens_8_8_max_abs_err":(out2-ref2).abs().max().item(),
 "metric":"max abs elementwise error vs same-sequence-restricted softmax reference; leaked-ref match identifies cross-sequence contamination",
 "shape":list(out.shape),"dtype":str(out.dtype)
}))
