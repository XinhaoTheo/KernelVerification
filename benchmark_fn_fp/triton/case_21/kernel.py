"""Triton kernel under test: fn13_paged_kv_block_table_homogeneous."""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_gather_kernel(KV, BlockTable, Out, max_blocks, page_size,
                         stride_kv_page, stride_out_seq,
                         HEAD_DIM: tl.constexpr):
    seq = tl.program_id(0)
    slot = tl.program_id(1)
    logical_block = slot // page_size
    within = slot % page_size

    physical = seq * max_blocks + logical_block
    offs = tl.arange(0, HEAD_DIM)
    src = KV + physical * stride_kv_page + within * HEAD_DIM + offs
    dst = Out + seq * stride_out_seq + slot * HEAD_DIM + offs
    tl.store(dst, tl.load(src))


def paged_gather(kv_cache: torch.Tensor, block_table: torch.Tensor,
                 seq_len: int) -> torch.Tensor:
    """Gather `seq_len` cached tokens for each sequence through its block table."""
    num_pages, page_size, head_dim = kv_cache.shape
    num_seqs, max_blocks = block_table.shape
    out = torch.empty((num_seqs, seq_len, head_dim), device=kv_cache.device,
                      dtype=kv_cache.dtype)
    _paged_gather_kernel[(num_seqs, seq_len)](
        kv_cache, block_table, out, max_blocks, page_size,
        kv_cache.stride(0), out.stride(0), HEAD_DIM=head_dim)
    return out
