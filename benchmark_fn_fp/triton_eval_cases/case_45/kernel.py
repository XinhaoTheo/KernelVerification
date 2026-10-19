"""Standalone routed expert projection adapted from vLLM's fused MoE kernel.

Copyright contributors to the vLLM project. Apache-2.0; see LICENSE.
2026 adaptation: one projection with explicit routing preparation and reduction,
fixed T4 launch configuration, executable input contract and FP64 reference.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _expert_projection(X, W, ROUTE_WEIGHT, SORTED, EXPERT, ROUTED,
                       K: tl.constexpr, N: tl.constexpr, TOPK: tl.constexpr,
                       R: tl.constexpr, EM: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    route = tl.load(SORTED + pid_m * BM + tl.arange(0, BM))
    valid = route < R
    expert = tl.load(EXPERT + pid_m)
    cols = pid_n * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + inner
        x = tl.load(X + (route[:, None] // TOPK) * K + kk[None, :],
                    mask=valid[:, None] & (kk[None, :] < K), other=0.0)
        w = tl.load(W + (expert * K + kk[:, None]) * N + cols[None, :],
                    mask=(kk[:, None] < K) & (cols[None, :] < N), other=0.0)
        accumulator = tl.dot(x, w, accumulator)
    weight = tl.load(ROUTE_WEIGHT + route, mask=valid, other=0.0)
    accumulator *= weight[:, None]
    tl.store(ROUTED + route[:, None] * N + cols[None, :], accumulator,
             mask=valid[:, None] & (cols[None, :] < N))


@triton.jit
def _combine_routes(ROUTED, OUT, M: tl.constexpr, N: tl.constexpr,
                    TOPK: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    value = tl.zeros((B,), tl.float32)
    for choice in range(TOPK):
        value += tl.load(ROUTED + (token * TOPK + choice) * N + cols,
                         mask=cols < N, other=0.0)
    tl.store(OUT + token * N + cols, value, mask=cols < N)


def validate_inputs(x, expert_weights, topk_ids, topk_weights, require_cuda=True):
    """Check the entire public domain; also usable for CPU reference inputs."""
    values = (x, expert_weights, topk_ids, topk_weights)
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    for tensor in values:
        require(type(tensor) is torch.Tensor, "standard torch.Tensor required")
        require(tensor.layout == torch.strided and tensor.is_contiguous(), "contiguous strided tensor required")
        require(not tensor.requires_grad and not tensor.is_inference(), "ordinary non-gradient tensors required")
        require(not tensor.is_neg() and not tensor.is_conj(), "materialized tensors without logical view flags required")
    require(all(t.device == x.device for t in values), "inputs must share one device")
    require(not require_cuda or x.is_cuda, "CUDA tensors required by run")
    require(x.ndim == 2 and expert_weights.ndim == 3 and topk_ids.ndim == 2,
            "expected x[M,K], weights[E,K,N], ids[M,Q]")
    m, k = x.shape
    experts, wk, n = expert_weights.shape
    q = topk_ids.shape[1]
    require(1 <= m <= 129 and 1 <= k <= 129 and 1 <= n <= 129,
            "M, K and N must be in [1,129]")
    require(1 <= experts <= 8 and 1 <= q <= min(4, experts), "E/Q outside supported domain")
    require(wk == k and topk_ids.shape[0] == m and topk_weights.shape == (m, q),
            "input shapes disagree")
    require(x.dtype == expert_weights.dtype == torch.float16, "x and expert weights must be FP16")
    require(topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32,
            "ids must be int32 and route weights FP32")
    for tensor in (x, expert_weights, topk_weights):
        require(bool(torch.isfinite(tensor).all().item()), "numeric inputs must be finite")
    require(bool((x.abs() <= .5).all().item()) and bool((expert_weights.abs() <= .5).all().item()),
            "x and expert-weight magnitudes must be at most 0.5")
    require(bool(((topk_ids >= 0) & (topk_ids < experts)).all().item()), "expert index out of range")
    require(bool(((topk_weights >= 0) & (topk_weights <= 1)).all().item()), "routing weights outside [0,1]")
    require(bool((topk_weights.double().sum(dim=1) <= 1).all().item()), "route weights sum exceeds one")
    if q > 1:
        ordered = topk_ids.sort(dim=1).values
        require(bool((ordered[:, 1:] != ordered[:, :-1]).all().item()), "each token must choose distinct experts")
    intervals = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in values]
    require(all(a1 <= b0 or b1 <= a0 for i, (a0, a1) in enumerate(intervals)
                for b0, b1 in intervals[i + 1:]), "input storage ranges must be disjoint")
    return {"valid": True, "tokens": m, "experts": experts, "topk": q,
            "input_width": k, "output_width": n}


def _prepare_routes(topk_ids, experts, block_m):
    ids = topk_ids.reshape(-1)
    order = torch.argsort(ids, stable=True)
    counts = torch.bincount(ids.long(), minlength=experts)
    padded = ((counts + block_m - 1) // block_m) * block_m
    starts = padded.cumsum(0) - padded
    ranks = torch.arange(ids.numel(), device=ids.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    slots = torch.repeat_interleave(starts, counts) + ranks
    size = int(padded.sum().item())
    sorted_ids = torch.full((size,), ids.numel(), device=ids.device, dtype=torch.int32)
    sorted_ids[slots] = order.to(torch.int32)
    expert_blocks = torch.repeat_interleave(torch.arange(experts, device=ids.device, dtype=torch.int32), padded // block_m)
    return sorted_ids, expert_blocks


def run(x, expert_weights, topk_ids, topk_weights):
    """Return a fresh FP32 [M,N] weighted expert projection; preserve inputs."""
    validate_inputs(x, expert_weights, topk_ids, topk_weights)
    m, k = x.shape
    experts, _, n = expert_weights.shape
    q = topk_ids.shape[1]
    with torch.cuda.device(x.device):
        sorted_ids, expert_blocks = _prepare_routes(topk_ids, experts, 16)
        routed = torch.zeros((m * q, n), device=x.device, dtype=torch.float32)
        output = torch.empty((m, n), device=x.device, dtype=torch.float32)
        grid = (expert_blocks.numel() * triton.cdiv(n, 32),)
        group_m = 4 if expert_blocks.numel() >= 4 else 1
        _expert_projection[grid](x, expert_weights, topk_weights, sorted_ids,
            expert_blocks, routed, K=k, N=n, TOPK=q, R=m*q, EM=sorted_ids.numel(),
            BM=16, BN=32, BK=32, GROUP_M=group_m, num_warps=4)
        _combine_routes[(m, triton.cdiv(n, 128))](routed, output, M=m, N=n,
                                              TOPK=q, B=128, num_warps=4)
    return output


def reference(x, expert_weights, topk_ids, topk_weights):
    """Independent token/choice FP64 target; no expert sorting or tile schedule."""
    validate_inputs(x, expert_weights, topk_ids, topk_weights, require_cuda=False)
    m, k = x.shape
    n = expert_weights.shape[2]
    result = torch.zeros((m, n), device=x.device, dtype=torch.float64)
    for choice in range(topk_ids.shape[1]):
        matrices = expert_weights[topk_ids[:, choice].long()].double()
        projected = torch.bmm(x.double().reshape(m, 1, k), matrices).squeeze(1)
        result += topk_weights[:, choice].double()[:, None] * projected
    return result


def make_inputs(device="cuda", tokens=37, experts=4, topk=2, input_width=33, output_width=65, seed=0):
    """One convenience generator, not a restriction on legal inputs."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand((tokens, input_width), generator=generator) - .5).half()
    weights = (torch.rand((experts, input_width, output_width), generator=generator) - .5).half()
    ids = torch.rand((tokens, experts), generator=generator).argsort(dim=1)[:, :topk].int().contiguous()
    route = torch.rand((tokens, topk), generator=generator)
    route = (route / (route.sum(dim=1, keepdim=True) + .125)).float()
    return x.to(device), weights.to(device), ids.to(device), route.to(device)
