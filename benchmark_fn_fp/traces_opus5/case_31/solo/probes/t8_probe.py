
import torch, json, importlib.util
spec = importlib.util.spec_from_file_location("kern", "/root/cases/case_31/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
dev='cuda'; d=16; H=1
qbs=16; kbs=16
cu_q=torch.tensor([0,8,32],device=dev,dtype=torch.int32)
cu_k=torch.tensor([0,8,32],device=dev,dtype=torch.int32)
S=32
q=torch.randn(S,H,d,device=dev,dtype=torch.float32)
k=torch.randn(S,H,d,device=dev,dtype=torch.float32)
v=torch.randn(S,H,d,device=dev,dtype=torch.float32)

nqb,cu_qb,qb2b,cu_nqb = m.calculate_blocks(cu_q,qbs)
nkb,cu_kb,kb2b,cu_nkb = m.calculate_blocks(cu_k,kbs)
info={"nqb":int(nqb),"cu_qb":cu_qb.tolist(),"cu_nqb":cu_nqb.tolist(),
      "nkb":int(nkb),"cu_kb":cu_kb.tolist(),"cu_nkb":cu_nkb.tolist()}

block_mask=torch.ones((H,nqb,nkb),device=dev,dtype=torch.bool)
out=m.block_sparse_attention(q,k,v,cu_q,cu_k,block_mask,qbs,kbs,causal=False,softmax_scale=None)

ref=torch.zeros_like(out); scale=d**-0.5
for b in range(2):
    qs,qe=cu_q[b].item(),cu_q[b+1].item()
    ks,ke=cu_k[b].item(),cu_k[b+1].item()
    for h in range(H):
        s=(q[qs:qe,h]@k[ks:ke,h].T)*scale
        ref[qs:qe,h]=torch.softmax(s,dim=-1)@v[ks:ke,h]

err=(out-ref).abs()
per_row=err.amax(dim=(1,2))

# "leaky" reference for seq0 rows: attends to keys 0..15 (crossing into seq1)
leak=torch.zeros(8,d,device=dev)
s=(q[0:8,0]@k[0:16,0].T)*scale
leak=torch.softmax(s,dim=-1)@v[0:16,0]
leak_err=(out[0:8,0]-leak).abs().max().item()

# second-sequence last partial block: keys 24..31 is exactly end of tensor -> no leak there
res={
 "info":info,
 "EVEN_SEQ_KBLOCK_flag": bool(((cu_k[-1]-cu_k[0])%kbs==0).item()),
 "per_seq_k_lens":[8,24],
 "max_abs_err_vs_contract_ref": err.max().item(),
 "max_err_seq0_rows(0-7)": per_row[0:8].max().item(),
 "max_err_seq1_rows(8-31)": per_row[8:].max().item(),
 "max_err_seq0_vs_LEAKY_ref(keys0..15)": leak_err,
 "metric":"elementwise max abs error of output vs per-sequence-restricted softmax reference; leaky ref shows kernel matches cross-sequence attention"
}
print(json.dumps(res))
