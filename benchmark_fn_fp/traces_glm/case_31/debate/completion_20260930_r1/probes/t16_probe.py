import torch, json, sys
sys.path.insert(0, "/root/cases/case_31")
import kernel as K

torch.manual_seed(0)
dev = 'cuda'
seqlens_q = [32, 32]
seqlens_k = [40, 24]   # total 64 -> EVEN_SEQ_KBLOCK=True (misclassified); batch0 last k block partial
cu_q = torch.tensor([0, 32, 64], device=dev, dtype=torch.int32)
cu_k = torch.tensor([0, 40, 64], device=dev, dtype=torch.int32)
nq, nk, hd = 1, 1, 64
q = torch.randn(64, nq, hd, device=dev)
k = torch.randn(64, nk, hd, device=dev)
v = torch.randn_like(k)
qb, kb = 32, 32
qb_per_batch = [(l + qb - 1)//qb for l in seqlens_q]
kb_per_batch = [(l + kb - 1)//kb for l in seqlens_k]
qblk_off = [0, qb_per_batch[0]]
kblk_off = [0, kb_per_batch[0]]
nqb, nkb = sum(qb_per_batch), sum(kb_per_batch)
bm = torch.ones(nk, nqb, nkb, dtype=torch.bool, device=dev)  # all selected

out = K.block_sparse_attention(q, k, v, cu_q, cu_k, bm, qb, kb, causal=False)

scale = hd ** -0.5
ref = torch.full_like(q, float('nan'))
for t in range(64):
    b = 0 if t < 32 else 1
    kstart, kend = int(cu_k[b]), int(cu_k[b+1])
    qbi = qblk_off[b] + (t - int(cu_q[b]))//qb
    cols = []
    for j in range(kb_per_batch[b]):
        if bm[0, qbi, kblk_off[b] + j]:
            ks = kstart + j*kb
            cols += list(range(ks, min(ks+kb, kend)))
    cols = torch.tensor(cols, device=dev)
    scores = (q[t,0].float() @ k[cols,0].float().T) * scale
    p = torch.softmax(scores, -1)
    ref[t,0] = p @ v[cols,0].float()

err = (out.float() - ref).abs()
# rows in batch 0 (query rows 0-31) attend batch0 keys [0,40); kernel may leak keys 40-63 (batch 1)
res = {
  "metric": "kernel output vs same-sequence-only masked-softmax reference (causal=False, all blocks selected)",
  "max_abs_err_batch0_rows": float(err[:32].max().item()),
  "max_abs_err_batch1_rows": float(err[32:].max().item()),
  "max_abs_err_overall": float(err.max().item()),
  "batch0_k_len": 40, "batch1_k_len": 24, "k_block_size": 32,
  "total_k_len": 64, "EVEN_SEQ_KBLOCK_kernel_value": True,
  "expected_signal": "large error on batch0 query rows if next-sequence keys leak into softmax"
}
print(json.dumps(res))
