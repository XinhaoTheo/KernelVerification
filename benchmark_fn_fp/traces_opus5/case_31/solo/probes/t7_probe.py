
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_31/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev='cuda'
d=16
# q seqlens [2,6] too so q blocks align; use q_block_size=4 too
q_seqlens=[2,6]; k_seqlens=[2,6]
qbs=4; kbs=4
cu_q=torch.tensor([0,2,8],device=dev,dtype=torch.int32)
cu_k=torch.tensor([0,2,8],device=dev,dtype=torch.int32)
H=1
q=torch.randn(8,H,d,device=dev,dtype=torch.float32)
k=torch.randn(8,H,d,device=dev,dtype=torch.float32)
v=torch.randn(8,H,d,device=dev,dtype=torch.float32)

nqb,cu_qb,qb2b,cu_nqb = m.calculate_blocks(cu_q,qbs)
nkb,cu_kb,kb2b,cu_nkb = m.calculate_blocks(cu_k,kbs)
print("nqb",nqb,"cu_qb",cu_qb.tolist(),"qb2b",qb2b.tolist(),"cu_nqb",cu_nqb.tolist())
print("nkb",nkb,"cu_kb",cu_kb.tolist(),"kb2b",kb2b.tolist(),"cu_nkb",cu_nkb.tolist())

block_mask=torch.ones((H,nqb,nkb),device=dev,dtype=torch.bool)
out=m.block_sparse_attention(q,k,v,cu_q,cu_k,block_mask,qbs,kbs,causal=False,softmax_scale=None)

# reference: dense per-sequence full attention (all blocks selected)
ref=torch.zeros_like(out)
scale=d**-0.5
for b in range(2):
    qs,qe=cu_q[b].item(),cu_q[b+1].item()
    ks,ke=cu_k[b].item(),cu_k[b+1].item()
    for h in range(H):
        s=(q[qs:qe,h]@k[ks:ke,h].T)*scale
        p=torch.softmax(s,dim=-1)
        ref[qs:qe,h]=p@v[ks:ke,h]

err=(out-ref).abs()
print(json.dumps({
 "max_abs_err": err.max().item(),
 "per_row_max_err": err.amax(dim=(1,2)).tolist(),
 "EVEN_SEQ_KBLOCK_flag": bool(((cu_k[-1]-cu_k[0])%kbs==0).item()),
 "per_seq_even": [ (l%kbs==0) for l in k_seqlens],
}))
