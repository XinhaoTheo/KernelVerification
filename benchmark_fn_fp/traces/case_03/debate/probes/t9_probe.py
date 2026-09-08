
import sys, json, numpy as np, torch
sys.path.insert(0,'/root/cases/case_03')
from kernel import gptq_matmul

M,K,N,gs,bits = 32,32,32,8,4
ng = K//gs
rng = np.random.default_rng(0)
q = rng.integers(0,16,size=(K,N)).astype(np.uint32)
packed = np.zeros((K//8,N),dtype=np.uint32)
for k in range(K):
    packed[k//8] |= (q[k] << ((k%8)*4))
packed_t = torch.from_numpy(packed.view(np.int32).copy()).cuda()
scales = torch.from_numpy((rng.random((ng,N))*0.1+0.05).astype(np.float32)).cuda()
zeros = torch.zeros((ng,N),dtype=torch.int32,device='cuda')   # all zero -> packed-unpack also yields 0
g_idx = (torch.arange(K,device='cuda')//gs).to(torch.int32)
a = torch.from_numpy(rng.standard_normal((M,K)).astype(np.float32)).cuda()

c = gptq_matmul(a, packed_t, scales, zeros, g_idx, bits=bits)
torch.cuda.synchronize()

qf = torch.from_numpy(q.astype(np.float32)).cuda()
sc_full = scales[g_idx.long()]                      # (K,N)
ref_contract = a @ ((qf - 0.0) * sc_full)           # problem.txt formula
ref_plus1    = a @ ((qf - 1.0) * sc_full)           # kernel's actual formula
predicted_offset = -(a @ sc_full)                   # (q-(z+1)) - (q-z) = -1 per element

e_contract = (c - ref_contract).abs().max().item()
e_plus1    = (c - ref_plus1).abs().max().item()
e_pred     = (c - (ref_contract + predicted_offset)).abs().max().item()
scale_c    = ref_contract.abs().max().item()

print(json.dumps({
  "shape": [M,K,N], "group_size": gs, "bits": bits,
  "zeros_all_zero": True,
  "ref_contract_absmax": scale_c,
  "max_abs_err_vs_contract_ref": e_contract,
  "rel_err_vs_contract_ref": e_contract/scale_c,
  "max_abs_err_vs_plus1_ref": e_plus1,
  "max_abs_err_vs_contract_plus_predicted_offset": e_pred,
  "matches_contract_within_1e-3": bool(e_contract < 1e-3),
  "matches_plus1_within_1e-3": bool(e_plus1 < 1e-3),
}))
