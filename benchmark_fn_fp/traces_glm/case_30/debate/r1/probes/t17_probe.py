import torch, json, sys
sys.path.insert(0, "/root/cases/case_31")
import kernel as K

torch.manual_seed(1)
dev = 'cuda'
# single batch, q len 32, k len 64, both multiples of block size -> isolates causal offset effect
seqlens_q = [32]; seqlens_k = [64]
cu_q = torch.tensor([0, 32], device=dev, dtype=torch.int32)
cu_k = torch.tensor([0, 64], device=dev, dtype=torch.int32)
nq, nk, hd = 1, 1, 64
q = torch.randn(32, nq, hd, device=dev)
k = torch.randn(64, nk, hd, device=dev)
v = torch.randn_like(k)
qb, kb = 32, 32
nqb, nkb = 1, 2
bm = torch.ones(nk, nqb, nkb, dtype=torch.bool, device=dev)

out = K.block_sparse_attention(q, k, v, cu_q, cu_k, bm, qb, kb, causal=True)

scale = hd ** -0.5
ref = torch.full_like(q, float('nan'))
for t in range(32):
    # position-based causal: query i attends key j iff j <= i (same sequence)
    cols = torch.arange(0, t+1, device=dev)
    scores = (q[t,0].float() @ k[cols,0].float().T) * scale
    p = torch.softmax(scores, -1)
    ref[t,0] = p @ v[cols,0].float()

err = (out.float() - ref).abs()
res = {
  "metric": "kernel output vs position-based causal reference (query i attends keys j<=i), causal=True, seq_q=32, seq_k=64",
  "max_abs_err": float(err.max().item()),
  "max_abs_err_row0": float(err[0].max().item()),
  "mean_abs_err": float(err.mean().item()),
  "seq_q": 32, "seq_k": 64, "offset_used_by_kernel": 32,
  "expected_signal": "large error if kernel's (k_len - q_len) offset shifts the causal boundary vs position-based causal"
}
print(json.dumps(res))
