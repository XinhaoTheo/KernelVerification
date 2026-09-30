import torch, sys, json
sys.path.insert(0, "/root/cases/case_21")
from kernel import paged_gather

torch.manual_seed(0)
num_seqs, max_blocks, page_size, head_dim = 2, 2, 4, 8
seq_len = page_size  # uses logical block 0 only; also test seq_len=page_size*2
num_pages = max(num_seqs * max_blocks, 4) + 1  # ensure non-identity mapping
kv = torch.randn(num_pages, page_size, head_dim, device="cuda")
# non-identity, interleaved mapping, arbitrary pages
block_table = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32, device="cuda")
ref = kv[block_table[:, None], torch.arange(seq_len, device="cuda")//page_size, torch.arange(seq_len, device="cuda")%page_size]
out = paged_gather(kv, block_table, seq_len)
mism = (out != ref).sum().item()
# also seq_len = 2 pages
seq_len2 = page_size * 2
out2 = paged_gather(kv, block_table, seq_len2)
t = torch.arange(seq_len2, device="cuda")
ref2 = kv[block_table[:, None].expand(num_seqs, seq_len2), t//page_size, t%page_size]
mism2 = (out2 != ref2).sum().item()
print(json.dumps({"block_table": block_table.tolist(), "mismatch_tokens_seq_len_4": mism,
                 "total_elems_1": out.numel(), "mismatch_elems_seq_len_8": mism2,
                 "total_elems_2": out2.numel(),
                 "max_abs_err_2": (out2.float()-ref2.float()).abs().max().item()}))