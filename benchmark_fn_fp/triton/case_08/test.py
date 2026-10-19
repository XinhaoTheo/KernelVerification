import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import sorted_topk_indices as route_buggy  # real, unmodified fla-org comparator

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


def route_reference(scores, k):
    import math
    B, N = scores.shape
    n_dims = int(math.log2(N))
    x = scores.clone().contiguous()
    ids = torch.arange(N, device=scores.device, dtype=torch.int32).unsqueeze(0).expand(B, N).contiguous()
    _row_argsort_kernel_ref[(B,)](x, ids, N, N=N, N_DIMS=n_dims, DESCENDING=True)
    return ids[:, :k].long()


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    torch.manual_seed(0)
    n_tokens, n_experts = 256, 4
    logits = torch.randn(n_tokens, n_experts, device=device)
    logits[10, 0] = 5.0
    logits[10, 2] = 5.0  # exact tie for token 10's top expert (this pair is verified to produce disagreement)

    expert_weight = torch.randn(n_experts, 8, device=device)

    ref_idx = route_reference(logits, k=1).squeeze(-1)
    cand_idx = route_buggy(logits, k=1).squeeze(-1)
    assert ref_idx[10].item() in (0, 2) and cand_idx[10].item() in (0, 2), "token 10's winner must come from the tied pair"
    assert ref_idx[10].item() != cand_idx[10].item(), "reference and real kernel should disagree on the tie"

    ref_out = expert_weight[ref_idx]
    cand_out = expert_weight[cand_idx]
    pooled_ref = ref_out.mean(dim=0)
    pooled_cand = cand_out.mean(dim=0)

    naive_pass = torch.allclose(pooled_cand, pooled_ref, rtol=1e-2, atol=1e-2)
    per_token_pass = torch.allclose(cand_out, ref_out, rtol=1e-2, atol=1e-2)
    print(f"token 10 tie: reference picked expert {ref_idx[10].item()}, real kernel picked expert {cand_idx[10].item()}")
    print(f"pooled diff = {(pooled_cand - pooled_ref).abs().max().item():.6f}, naive allclose on pooled mean = {naive_pass}")
    print(f"per-token allclose (would catch it) = {per_token_pass}")
    print("FN DEMONSTRATED" if naive_pass and not per_token_pass else "tune constants")
    # What a conventional CI test would conclude: pooled-mean allclose，随机分数几乎不产生 tie
    print(f"NAIVE_ALLCLOSE_VERDICT: {naive_pass}")
    return naive_pass


if __name__ == "__main__":
    test_kernel()
