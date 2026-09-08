
import importlib.util, json, torch

spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_31/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

dev = "cuda"
torch.manual_seed(0)

def blocks(cu, bs):
    cun = [0]
    for b in range(len(cu)-1):
        L = cu[b+1]-cu[b]
        cun.append(cun[-1] + -(-L//bs))
    return cun

def ref_attn(q,k,v,cu_q,cu_k,mask,qbs,kbs,scale,hsel):
    out = torch.zeros_like(q)
    B = len(cu_q)-1
    cu_nq = blocks(cu_q,qbs); cu_nk = blocks(cu_k,kbs)
    Hq = q.shape[1]
    for b in range(B):
        qs,qe = cu_q[b],cu_q[b+1]; ks,ke = cu_k[b],cu_k[b+1]
        Lq,Lk = qe-qs, ke-ks
        nqb = -(-Lq//qbs); nkb = -(-Lk//kbs)
        for hq in range(Hq):
            hm = hsel(hq)
            allowed = torch.zeros(Lq,Lk,dtype=torch.bool,device=dev)
            for i in range(nqb):
                r0,r1 = i*qbs, min((i+1)*qbs,Lq)
                for j in range(nkb):
                    c0,c1 = j*kbs, min((j+1)*kbs,Lk)
                    if bool(mask[hm, cu_nq[b]+i, cu_nk[b]+j]):
                        allowed[r0:r1,c0:c1] = True
            hk = hq // (q.shape[1]//k.shape[1])
            s = q[qs:qe,hq].float() @ k[ks:ke,hk].float().T * scale
            s = s.masked_fill(~allowed, float('-inf'))
            p = torch.nan_to_num(torch.softmax(s,-1), nan=0.0)
            out[qs:qe,hq] = (p @ v[ks:ke,hk].float()).to(out.dtype)
    return out

D = 16
qlen = [3,5]; klen = [3,5]
qbs, kbs = 4, 4
cu_q = [0,3,8]; cu_k = [0,3,8]
Sq, Sk = 8, 8
H = 1
q = torch.randn(Sq,H,D,device=dev,dtype=torch.float32)
k = torch.randn(Sk,H,D,device=dev,dtype=torch.float32)
v = torch.randn(Sk,H,D,device=dev,dtype=torch.float32)
cu_q_t = torch.tensor(cu_q,device=dev,dtype=torch.int32)
cu_k_t = torch.tensor(cu_k,device=dev,dtype=torch.int32)
nqb = blocks(cu_q,qbs)[-1]; nkb = blocks(cu_k,kbs)[-1]
mask = torch.ones(H,nqb,nkb,device=dev,dtype=torch.bool)
scale = D ** -0.5

flag_k = bool(((cu_k_t[-1]-cu_k_t[0]) % kbs == 0).item())
flag_q = bool(torch.all((cu_q_t[1:]-cu_q_t[:-1]) % qbs == 0).item())

out = m.block_sparse_attention(q,k,v,cu_q_t,cu_k_t,mask,qbs,kbs,False,None)
ref = ref_attn(q,k,v,cu_q,cu_k,mask,qbs,kbs,scale,lambda h:h)
err = (out-ref).abs()

# "leaked" reference for sequence 0: its single partial k block spans rows 0..3 (row 3 belongs to seq 1)
s_leak = q[0:3,0].float() @ k[0:4,0].float().T * scale
p_leak = torch.softmax(s_leak,-1)
leak_out = p_leak @ v[0:4,0].float()
leak_err = (out[0:3,0]-leak_out).abs().max().item()

# control: truly even key lengths [4,4]
cu_k2 = [0,4,8]
cu_k2_t = torch.tensor(cu_k2,device=dev,dtype=torch.int32)
cu_q2 = [0,4,8]; cu_q2_t = torch.tensor(cu_q2,device=dev,dtype=torch.int32)
nqb2 = blocks(cu_q2,qbs)[-1]; nkb2 = blocks(cu_k2,kbs)[-1]
mask2 = torch.ones(H,nqb2,nkb2,device=dev,dtype=torch.bool)
out2 = m.block_sparse_attention(q,k,v,cu_q2_t,cu_k2_t,mask2,qbs,kbs,False,None)
ref2 = ref_attn(q,k,v,cu_q2,cu_k2,mask2,qbs,kbs,scale,lambda h:h)
err2 = (out2-ref2).abs().max().item()

print(json.dumps({
 "EVEN_SEQ_KBLOCK_flag_lens_3_5": flag_k,
 "EVEN_SEQ_QBLOCK_flag": flag_q,
 "max_abs_err_vs_same_seq_ref": err.max().item(),
 "max_abs_err_seq0_rows": err[0:3].max().item(),
 "max_abs_err_seq1_rows": err[3:8].max().item(),
 "seq0_err_vs_leaked_ref": leak_err,
 "control_even_lens_4_4_max_abs_err": err2,
 "metric": "max abs elementwise error vs same-sequence-restricted softmax reference (continuous output)",
 "shape": list(out.shape), "dtype": str(out.dtype)
}))
