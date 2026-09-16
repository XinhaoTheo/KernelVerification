
import torch, json, importlib.util, sys
spec = importlib.util.spec_from_file_location("k", "/root/cases/case_21/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
num_pages, page_size, head_dim = 16, 4, 8
num_seqs, max_blocks = 4, 4
seq_len = 8
kv = torch.randn(num_pages, page_size, head_dim, device='cuda')
perm = torch.randperm(num_pages, device='cuda')[:num_seqs*max_blocks].to(torch.int32)
bt = perm.view(num_seqs, max_blocks).contiguous()

out = m.paged_gather(kv, bt, seq_len)
t = torch.arange(seq_len, device='cuda')
ref = kv[bt[:, (t//page_size)].long(), (t % page_size).unsqueeze(0).expand(num_seqs, seq_len)]
maxerr = (out-ref).abs().max().item()
# identity-contiguous table check
bt_id = torch.arange(num_seqs*max_blocks, device='cuda', dtype=torch.int32).view(num_seqs, max_blocks)
out_id = m.paged_gather(kv, bt_id, seq_len)
ref_id = kv[bt_id[:, (t//page_size)].long(), (t % page_size).unsqueeze(0).expand(num_seqs, seq_len)]
err_id = (out_id-ref_id).abs().max().item()
print(json.dumps({"metric":"max_abs_err_vs_contract_formula",
 "block_table": bt.tolist(),
 "max_err_permuted": maxerr,
 "rows_mismatched": int((out!=ref).any(-1).sum().item()),
 "total_rows": num_seqs*seq_len,
 "max_err_contiguous_identity_table": err_id}))
