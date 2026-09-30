import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import sorted_topk_indices as select_topk_buggy  # real, unmodified fla-org comparator

# Reference: the UNMODIFIED real NSA sort kernel, defined inline so this file
# doesn't depend on kernel.py's mutation.
import triton
import triton.language as tl


@triton.jit
def _compare_and_swap_ref(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
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
    cond = (left >= right) != flip  # enforces this benchmark's lower-index-wins tie contract
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge_ref(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap_ref(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def _row_argsort_kernel_ref(X_ptr, IDS_ptr, stride_row, N: tl.constexpr, N_DIMS: tl.constexpr, DESCENDING: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(X_ptr + row * stride_row + offs)
    ids = tl.load(IDS_ptr + row * stride_row + offs)
    for i in tl.static_range(1, N_DIMS + 1):
        x, ids = _bitonic_merge_ref(x, ids, i, 2 if i < N_DIMS else DESCENDING, N_DIMS)
    tl.store(X_ptr + row * stride_row + offs, x)
    tl.store(IDS_ptr + row * stride_row + offs, ids)


def select_topk_reference(scores, k):
    import math
    B, N = scores.shape
    n_dims = int(math.log2(N))
    x = scores.clone().contiguous()
    ids = torch.arange(N, device=scores.device, dtype=torch.int32).unsqueeze(0).expand(B, N).contiguous()
    _row_argsort_kernel_ref[(B,)](x, ids, N, N=N, N_DIMS=n_dims, DESCENDING=True)
    return ids[:, :k].long()


def run_downstream(scores, values, k, select_fn):
    idx = select_fn(scores, k)
    gathered = torch.gather(
        values.unsqueeze(0).expand(scores.shape[0], -1, -1),
        1,
        idx.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
    )
    return gathered.mean(dim=1)


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    torch.manual_seed(0)
    n, k, d = 32, 8, 16
    # Deterministic construction: 7 indices with strictly distinct, clearly
    # higher scores fill ranks 0-6; indices 7 and 20 are tied for the 8th
    # (last) slot; every other index is clearly lower. This avoids any
    # accidental extra tie that random scores + a copied boundary value could
    # create.
    scores = torch.full((1, n), -100.0, device=device)
    higher_idxs = [0, 1, 2, 3, 4, 5, 6]
    for rank, idx in enumerate(higher_idxs):
        scores[0, idx] = 100.0 - rank
    scores[0, 7] = 50.0
    scores[0, 20] = 50.0  # exact tie at the k-th boundary between idx 7 and idx 20

    values = torch.randn(n, d, device=device) * 0.3
    values[20] = values[7] + torch.randn(d, device=device) * 0.03  # diluted swap

    ref_idx = select_topk_reference(scores, k)
    cand_idx = select_topk_buggy(scores, k)
    ref_tie_pick = ({7, 20} & set(ref_idx[0].tolist()))
    cand_tie_pick = ({7, 20} & set(cand_idx[0].tolist()))
    # Bitonic-network tie-break direction depends on the bit patterns of the
    # specific tied positions, not simply "lower index" in the abstract -- so
    # we verify empirically that the contract-enforcing reference and the
    # real kernel's default actually disagree on this tie, rather than
    # presupposing which physical index each one keeps.
    assert len(ref_tie_pick) == 1 and len(cand_tie_pick) == 1, "each should keep exactly one of the tied pair"
    assert ref_tie_pick != cand_tie_pick, "reference and real kernel should disagree on the tie"

    ref_out = run_downstream(scores, values, k, select_topk_reference)
    cand_out = run_downstream(scores, values, k, select_topk_buggy)

    naive_pass = torch.allclose(cand_out, ref_out, rtol=1e-2, atol=1e-2)
    max_abs = (cand_out - ref_out).abs().max().item()
    print(f"real NSA kernel (GPU): contract-enforcing reference keeps idx {list(ref_tie_pick)[0]}, real unmodified kernel keeps idx {list(cand_tie_pick)[0]} on the tie")
    print(f"downstream max abs diff = {max_abs:.6f}, naive allclose = {naive_pass}")
    print("FN DEMONSTRATED" if naive_pass else "tune constants")
    # What a conventional CI test would conclude: 下游 allclose，同上
    print(f"NAIVE_ALLCLOSE_VERDICT: {naive_pass}")
    return naive_pass


if __name__ == "__main__":
    test_kernel()
