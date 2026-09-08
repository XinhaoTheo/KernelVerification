
import sys, json, numpy as np, torch
sys.path.insert(0,'/root/cases/case_03')
from kernel import gptq_matmul

M,K,N,gs,bits = 32,32,32,8,4
ng = K//gs
rng = np.random.default_rng(1)
q = rng.integers(0,16,size=(K,N)).astype(np.uint32)
packed = np.zeros((K//8,N),dtype=np.uint32)
for k in range(K):
    packed[k//8] |= (q[k] << ((k%8)*4))
packed_t = torch.from_numpy(packed.view(np.int32).copy()).cuda()
scales = torch.from_numpy((rng.random((ng,N))*0.1+0.05).astype(np.float32)).cuda()
zeros_np = rng.integers(1,16,size=(ng,N)).astype(np.int32)   # per-(group,column) zero point, declared layout
zeros = torch.from_numpy(zeros_np).cuda()
g_idx = (torch.arange(K,device='cuda')//gs).to(torch.int32)
a = torch.from_numpy(rng.standard_normal((M,K)).astype(np.float32)).cuda()

c = gptq_matmul(a, packed_t, scales, zeros, g_idx, bits=bits)
torch.cuda.synchronize()

qf = torch.from_numpy(q.astype(np.float32)).cuda()
gl = g_idx.long()
sc_full = scales[gl]                                  # (K,N)
z_col   = zeros[gl].float()                           # (K,N) declared per-column zeros

ref_contract = a @ ((qf - z_col) * sc_full)                 # problem.txt formula
ref_plus1    = a @ ((qf - (z_col + 1.0)) * sc_full)         # only the c1 bug applied

# emulate the kernel's packed-along-N zeros access
n = np.arange(N)
z_packed_np = (zeros_np[:, n//8] >> ((n % 8) * 4)) & 15     # (ng,N)
z_packed = torch.from_numpy(z_packed_np.astype(np.float32)).cuda()[gl]
ref_kernel_semantics = a @ ((qf - (z_packed + 1.0)) * sc_full)

def m(x): return (c - x).abs().max().item()
scale_c = ref_contract.abs().max().item()

col_err = (c - ref_plus1).abs().max(dim=0).values          # per-column err vs "+1 only" ref
aligned = col_err[torch.arange(N, device=c.device) % 8 == 0]
misaligned = col_err[torch.arange(N, device=c.device) % 8 != 0]

print(json.dumps({
  "shape": [M,K,N], "group_size": gs, "bits": bits,
  "ref_contract_absmax": scale_c,
  "max_abs_err_vs_contract_ref": m(ref_contract),
  "max_abs_err_vs_plus1_only_ref": m(ref_plus1),
  "max_abs_err_vs_packed_zeros_kernel_semantics": m(ref_kernel_semantics),
  "rel_err_vs_plus1_only_ref": m(ref_plus1)/scale_c,
  "percol_err_vs_plus1_ref_n_mod8_eq0_max": aligned.max().item(),
  "percol_err_vs_plus1_ref_n_mod8_ne0_min": misaligned.min().item(),
  "n_cols_with_err_gt_1e-3_vs_plus1_ref": int((col_err > 1e-3).sum().item()),
  "N": N,
  "matches_kernel_packed_semantics_within_1e-3": bool(m(ref_kernel_semantics) < 1e-3),
}))
