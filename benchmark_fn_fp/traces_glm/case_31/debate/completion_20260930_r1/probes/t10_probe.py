
import torch, json, math, sys
sys.path.insert(0, "/root/cases/case_31")
import kernel as K

torch.manual_seed(0)
dev = 'cuda'
seqlens_q = [40, 40]
seqlens_k = [64, 64]
cu_q = torch.tensor([0]+[sum(seqlens_q[:i+1]) for i in range(2)], device=dev, dtype=torch.int32)
cu_k = torch.tensor([0]+[sum(seqlens_k[:i+1]) for i in range(2)], device=dev, dtype=torch.int32)
nq, nk, hd = 1, 1, 64
q = torch.randn(cu_q[-1].item(), nq, hd, device=dev)
k = torch.randn(cu_k[-1].item(), nk, hd, device=dev)
v = torch.randn_like(k)
qb, kb = 32, 32

# q-block id for a global row t (batch-local block index + global q-block offset)
qb_per_batch = [ (l + qb - 1)//qb for l in seqlens_q ]
kb_per_batch = [ (l + kb - 1)//kb for l in seqlens_k ]
qblk_offsets = [0, qb_per_batch[0]]  # global q-block start per batch
kblk_offsets = [0, kb_per_batch[0]]
nqb, nkb = sum(qb_per_batch), sum(kb_per_batch)

bm = torch.ones(nk, nqb, nkb, dtype=torch.bool, device=dev)
bm[0, 0, :] = False  # query block 0 of batch 0 has no selected k block

out = K.block_sparse_attention(q, k, v, cu_q, cu_k, bm, qb, kb, causal=False)

scale = hd ** -0.5
ref = torch.full_like(q, float('nan'))
for t in range(q.shape[0]):
    b = 0 if t < seqlens_q[0] else 1
    qstart, qend = int(cu_q[b]), int(cu_q[b+1])
    kstart, kend = int(cu_k[b]), int(cu_k[b+1])
    qbi = qblk_offsets[b] + (t - qstart)//qb
    valid_js = [j for j in range(kb_per_batch[b]) if bm[0, qbi, kblk_offsets[b] + j]]
    if not valid_js:
        continue  # empty key set -> reference undefined (NaN placeholder)
    cols = []
    for j in valid_js:
        ks = kstart + j*kb
        cols += list(range(ks, min(ks+kb, kend)))
    cols = torch.tensor(cols, device=dev)
    scores = (q[t,0].float() @ k[cols,0].float().T) * scale
    p = torch.softmax(scores, -1)
    ref[t,0] = p @ v[cols,0].float()

empty_rows = torch.arange(0, 32, device=dev)
other_rows = torch.arange(32, 80, device=dev)
res = {
  "metric": "kernel output vs masked-softmax reference; empty-key rows reported separately since reference is undefined there",
  "kernel_out_abs_max_empty_rows": float(out[empty_rows].abs().max().item()),
  "kernel_out_empty_rows_all_zero": bool((out[empty_rows] == 0).all().item()),
  "kernel_out_all_finite": bool(torch.isfinite(out.float()).all().item()),
  "ref_empty_rows": "NaN (undefined: empty key set; natural torch softmax over empty set would be NaN)",
  "max_abs_err_nonempty_rows": float((out[other_rows]-ref[other_rows]).abs().max().item()),
  "note": "kernel emits exact zeros on empty-key rows; reference behavior for empty softmax rows is undefined by problem.txt, so match/mismatch is undecidable from this probe alone."
}
print(json.dumps(res))
