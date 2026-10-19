"""Triton kernel under test: fn1_nsa_bitonic_topk_tie_dilution."""
import torch
import triton
import triton.language as tl

@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def _row_argsort_kernel(X_ptr, IDS_ptr, stride_row, N: tl.constexpr, N_DIMS: tl.constexpr, DESCENDING: tl.constexpr):
    # tl.log2(x.shape[dim]); that call path is not constexpr-stable on every
    # Triton version, so N_DIMS is passed in from the (Python-side, so still
    # compile-time-known) caller instead. The bitonic merge/compare-and-swap
    # primitives below -- where the actual tie-break mechanism lives -- are
    # byte-for-byte the real kernel, unchanged.
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(X_ptr + row * stride_row + offs)
    ids = tl.load(IDS_ptr + row * stride_row + offs)
    for i in tl.static_range(1, N_DIMS + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < N_DIMS else DESCENDING, N_DIMS)
    tl.store(X_ptr + row * stride_row + offs, x)
    tl.store(IDS_ptr + row * stride_row + offs, ids)


def sorted_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Host wrapper: real bitonic-sort kernel, launched one program per row."""
    import math
    assert scores.is_cuda
    B, N = scores.shape
    n_dims = int(math.log2(N))
    x = scores.clone().contiguous()
    ids = torch.arange(N, device=scores.device, dtype=torch.int32).unsqueeze(0).expand(B, N).contiguous()
    _row_argsort_kernel[(B,)](x, ids, N, N=N, N_DIMS=n_dims, DESCENDING=True)
    return ids[:, :k].long()
