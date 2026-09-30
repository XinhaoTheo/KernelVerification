import torch
from kernel import paged_gather


def reference(kv_cache, block_table, seq_len):
    page_size = kv_cache.shape[1]
    num_seqs = block_table.shape[0]
    out = torch.empty((num_seqs, seq_len, kv_cache.shape[2]), device=kv_cache.device,
                      dtype=kv_cache.dtype)
    for s in range(num_seqs):
        for t in range(seq_len):
            out[s, t] = kv_cache[block_table[s, t // page_size].item(), t % page_size]
    return out


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    num_pages, page_size, head_dim = 32, 8, 16
    num_seqs, max_blocks, seq_len = 4, 4, 32

    # --- test data ------------------------------------------------------------
    kv = torch.randn(num_pages, page_size, head_dim, device=device)
    kv += torch.arange(num_pages, device=device).view(-1, 1, 1) * 10.0

    # Lazy test data: pages happen to be handed out contiguously in order.
    identity = torch.arange(num_seqs * max_blocks, device=device,
                            dtype=torch.int32).reshape(num_seqs, max_blocks)
    cand_h = paged_gather(kv, identity, seq_len)
    ref_h = reference(kv, identity, seq_len)
    homogeneous_pass = torch.allclose(cand_h, ref_h, rtol=1e-2, atol=1e-2)

    # Contract-permitted table: pages of different sequences are interleaved,
    # which is what a real allocator produces once requests come and go.
    perm = torch.randperm(num_pages, device=device)[:num_seqs * max_blocks]
    shuffled = perm.to(torch.int32).reshape(num_seqs, max_blocks)
    cand_d = paged_gather(kv, shuffled, seq_len)
    ref_d = reference(kv, shuffled, seq_len)
    distinct_pass = torch.allclose(cand_d, ref_d, rtol=1e-2, atol=1e-2)
    diff = (cand_d - ref_d).abs().max().item()

    print(f"pages handed out in order: allclose={homogeneous_pass}")
    print(f"pages interleaved by the allocator: allclose={distinct_pass}, max abs diff={diff:.4f}")
    # What a conventional CI test would conclude: it fills the cache with one
    # repeated pattern, so reading the wrong page returns the same values.
    print(f"NAIVE_ALLCLOSE_VERDICT: {homogeneous_pass}")
    return homogeneous_pass and not distinct_pass


if __name__ == "__main__":
    test_kernel()
