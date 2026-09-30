import torch, sys, importlib.util
spec = importlib.util.spec_from_file_location("kernel", "/root/cases/case_21/kernel.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(0)
num_pages, page_size, head_dim = 16, 4, 8
num_seqs, max_blocks = 3, 5
kv = torch.randn(num_pages, page_size, head_dim, device="cuda")
# permuted (non-identity) block table, distinct pages
perm = torch.randperm(num_pages)[:num_seqs*max_blocks] if num_pages >= num_seqs*max_blocks else torch.randperm(num_pages).repeat(2)
bt = perm[:num_seqs*max_blocks].reshape(num_seqs, max_blocks).to(torch.int32).cuda()
seq_len = 13  # fits within max_blocks pages

out = m.paged_gather(kv, bt, seq_len)

# contract reference
t = torch.arange(seq_len, device="cuda")
phys = bt[:, t // page_size]  # [num_seqs, seq_len]
ref = kv[phys, t % page_size]

mism = (out != ref).any(dim=-1)
print(torch.cuda.is_available())
print({"max_abs_err": (out - ref).abs().max().item(),
       "mismatch_frac": mism.float().mean().item()})
# also identity-mapping sanity check
bt_ident = torch.arange(num_seqs*max_blocks, dtype=torch.int32, device="cuda").reshape(num_seqs, max_blocks)
kv2 = torch.randn(num_seqs*max_blocks, page_size, head_dim, device="cuda")
out2 = m.paged_gather(kv2, bt_ident, seq_len)
phys2 = bt_ident[:, t // page_size]
ref2 = kv2[phys2, t % page_size]
print("identity_ok", bool(torch.allclose(out2, ref2)))
