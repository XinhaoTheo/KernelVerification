import torch, sys, json
sys.path.insert(0, "/root/cases/case_21")
from kernel import paged_gather

torch.manual_seed(0)
num_seqs, max_blocks, page_size, head_dim = 2, 2, 4, 8
num_pages = 8
kv = torch.randn(num_pages, page_size, head_dim, device="cuda")
block_table = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32, device="cuda")

def ref_gather(seq_len):
    t = torch.arange(seq_len, device="cuda")
    out = torch.empty(num_seqs, seq_len, head_dim, device="cuda")
    for s in range(num_seqs):
        for i in range(seq_len):
            out[s, i] = kv[int(block_table[s, i // page_size]), i % page_size]
    return out

res = {"block_table": block_table.tolist()}
for seq_len in (page_size, page_size * 2):
    ref = ref_gather(seq_len)
    out = paged_gather(kv, block_table, seq_len)
    res[f"mismatch_elems_seq_len_{seq_len}"] = int((out != ref).sum().item())
    res[f"total_elems_seq_len_{seq_len}"] = out.numel()
    res[f"max_abs_err_seq_len_{seq_len}"] = (out.float() - ref.float()).abs().max().item()
print(json.dumps(res))