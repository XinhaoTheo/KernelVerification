
import torch, json, math
import importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_31/kernel.py")
# fall back: load via sys.path
import sys
sys.path.insert(0, "/root/cases/case_31")
import kernel as K

torch.manual_seed(0)
dev = 'cuda'
BS = 2
seqlens_q = [40, 40]
seqlens_k = [64, 64]
cu_q = torch.tensor([0]+list(torch.cumsum(torch.tensor(seqlens_q),0)), device=dev, dtype=torch.int32)
cu_k = torch.tensor([0]+list(torch.cumsum(torch.tensor(seqlens_k),0)), device=dev, dtype=torch.int32)
nq, nk, hd = 1, 1, 64
q = torch.randn(cu_q[-1].item(), nq, hd, device=dev)
k = torch.randn(cu_k[-1].item(), nk, hd, device=dev)
v = torch.randn_like(k)
qb, kb = 32, 32
# compute block counts
def blocks(cu, bs):
    sizes = (cu[1:]-cu[:-1])
    nb = (sizes + bs - 1)//bs
    return int(nb.sum().item())
nqb = blocks(cu_q, qb)
nkb = blocks(cu_k, kb)
# block_mask: all True EXCEPT query block 0 of batch 0 has no selected k block
bm = torch.ones(nk, nqb, nkb, dtype=torch.bool, device=dev)
# q blocks per batch: batch0 -> blocks 0,1 ; batch1 -> blocks 2,3
bm[0, 0, :] = False

out = K.block_sparse_attention(q, k, v, cu_q, cu_k, bm, qb, kb, causal=False)

# reference: per-row masked softmax over in-scope keys
scale = hd ** -0.5
ref = torch.full_like(q, float('nan'))
rows = cu_q[0].item()  # first 32 rows are the empty rows
for t in range(q.shape[0]):
    b = int((cu_q <= t).sum().item()) - 1
    qstart, qend = cu_q[b].item(), cu_q[b+1].item()
    kstart, kend = cu_k[b].item(), cu_k[b+1].item()
    # which q block / k blocks selected
    qbi = (t - qstart)//qb
    valid_js = [j for j in range((kend-kstart+kb-1)//kb) if bm[0, int(((cu_q[1:]-cu_q[:-1])[:b].sum().item())) + qbi, j]]
    if not valid_js:
        continue  # leave NaN (undefined)
    cols = []
    for j in valid_js:
        ks = kstart + j*kb
        cols += list(range(ks, min(ks+kb, kend)))
    cols = torch.tensor(cols, device=dev)
    scores = (q[t,0].float() @ k[cols,0].float().T) * scale
    p = torch.softmax(scores, -1)
    ref[t,0] = (p @ v[cols,0].float()).to(q.dtype)

empty_rows = torch.arange(0, 32, device=dev)
other_rows = torch.arange(32, 80, device=dev)
res = {
  "kernel_out_abs_max_empty_rows": float(out[empty_rows].abs().max().item()),
  "kernel_out_all_finite": bool(torch.isfinite(out.float()).all().item()),
  "ref_empty_rows": "NaN (undefined: empty key set)",
  "max_abs_err_nonempty_rows": float((out[other_rows]-ref[other_rows]).abs().max().item()),
  "note": "kernel produces exact zeros on empty rows; reference softmax over empty set is mathematically undefined (0/0); natural torch masked-softmax gives NaN. No reference implementation for empty rows in artifact, so match is undecidable from this probe."
}
print(json.dumps(res))
