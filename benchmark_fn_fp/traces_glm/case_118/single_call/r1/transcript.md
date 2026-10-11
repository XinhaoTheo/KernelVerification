# Single-call trace

## System prompt

You are verifying whether a GPU (Triton) kernel implementation is correct.

You will be given:
1. A problem statement describing the operation's contract and its reference behavior.
2. The kernel implementation under test.

Decide one of:
- "reject": the implementation has a real defect / violates the stated contract.
- "trust": the implementation is correct, or is an equally valid alternative implementation.
- "needs_more_evidence": you genuinely cannot tell.

Answer with the JSON object required by the output schema.
Output schema:
{"type": "object", "required": ["verdict", "confidence", "reason"], "properties": {"verdict": {"type": "string", "enum": ["trust", "reject", "needs_more_evidence"], "description": "'reject' if the implementation has a real defect, 'trust' if it is correct or an equally valid alternative, 'needs_more_evidence' if you genuinely cannot tell."}, "confidence": {"type": "number", "description": "0.0 to 1.0."}, "reason": {"type": "string", "description": "One or two sentences."}}, "additionalProperties": false}

## User prompt

## Problem statement (contract + reference behavior)

Compute a routed mixture-of-experts linear projection. Each token selects Q
experts; the result is the weighted sum of their projections. This is the
projection stage, not a complete nonlinear MoE layer or a router/top-k search.

Inputs:
- x: FP16 [M,K], finite, magnitude at most 0.5.
- expert_weights: FP16 [E,K,N], finite, magnitude at most 0.5.
- topk_ids: int32 [M,Q], each entry in [0,E). The Q choices within one token
  are distinct; choices may occur in any order. Tokens may share experts;
  an expert may receive zero routes or any number of routes up to M.
- topk_weights: FP32 [M,Q], finite, each in [0,1], and each row's sum
  evaluated in FP64 at most one. Zero weights and all-zero rows are legal.
- 1 <= M,K,N <= 129, 1 <= E <= 8, 1 <= Q <= min(4,E).

All inputs are ordinary torch.Tensor objects (not subclasses), dense, strided,
contiguous and on one CUDA device, with disjoint storage ranges. Inputs have
requires_grad=False, is_inference=False, is_neg=False and is_conj=False. Calls
execute in ordinary non-inference mode with no concurrent external input writes.
The interface handles materialized numerical buffers and does not provide autograd.

In real arithmetic on the stored input values, define:
  target[m,n] = sum over q=0..Q-1 of
    topk_weights[m,q] * sum over k=0..K-1 of
      x[m,k] * expert_weights[topk_ids[m,q],k,n].
Return a fresh finite FP32 CUDA tensor of shape [M,N]. Preserve every input.
Every output must satisfy
  abs(actual - target) <= 0.01 + 0.0002 * abs(target).
Input and weight bounds, row weight sums and K imply |target| < 32.251
(including the tiny rounding margin of the FP64 row-sum check).
This tolerance accommodates the bounded FP16-input/FP32-accumulation path,
including conservative allowance for small FP16 contributions. It does not
permit missing routes, output tiles or expert projections.

Grouping, expert order, padding and tile sizes are implementation choices and
do not change the target. Permuting each token's choices together with its
weights, or consistently relabeling experts and their matrices, preserves the
required result. Correctness covers every legal shape, value and routing layout;
there is one verdict for this fixed implementation, not a verdict per seed.

validate_inputs() enforces this public domain and is called by run(). reference()
is an independent FP64 implementation and accepts CPU or CUDA inputs.
make_inputs() is only a convenience example; it does not define the input domain.
Frozen environment: NVIDIA T4, PyTorch 2.8.0, Triton 3.4.0.


## Kernel implementation under test (kernel.py)

```python
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

```

Is this implementation correct?

## Provider reasoning (verbatim)

Analyze the kernel. Key concern: pid_n computation when group_size_m < GROUP_M — standard vLLM grid uses num_pid_in_group = GROUP_M * num_pid_n and group_size_m = min(...). The typical formula: pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m; pid_n = (pid % num_pid_in_group) // group_size_m. That's vLLM's standard. Fine.

Wait — grid is expert_blocks.numel() * cdiv(n,32) programs, but grouped ordering assumes total pid_m = EM/BM blocks. num_pid_m = cdiv(EM,BM). Since EM is padded to multiples of BM=16, num_pid_m = padded_blocks count = expert_blocks.numel(). Good.

Sorting: sorted_ids holds original flat route indices sorted by expert, padded with sentinel ids.numel(). valid = route < R = m*q. Sentinel is m*q, so masked. Good.

X load: route // TOPK gives token index. Good. ROUTE_WEIGHT loaded at route index into flat [M*Q] FP32 — topk_weights contiguous, ok. Weight applied per row of block. Store ROUTED + route*N + cols: each route writes its own row of routed. Since experts are distinct per token but tokens can share experts — within a block, could two routes with same expert... store targets distinct rows (route index distinct), fine. tl.store with masks.

Potential race: different blocks write distinct routes? Each route appears once in sorted_ids, so no duplicates. Good.

Ranks computation: arange(numel) - repeat_interleave(cumsum-counts, counts) gives rank within each expert group. slots = starts[expert] + rank. starts from padded cumsum. Correct.

expert_blocks: repeat_interleave(arange(experts), padded//16) — number of blocks per expert. Grid covers all pid_m blocks × pid_n.

_combine_routes: sums TOPK rows of routed per token, correct. output fresh tensor.

Edge case: R = m*q; sentinel m*q; valid route < R masks sentinel. tl.load of SORTED + pid_m*BM + arange(BM) — pid_m*BM up to EM-1+... EM is multiple of 16, load within bounds since pid_m < num_pid_m, pid_m*BM+BM <= EM. Good.

Small case: M=37, Q=2, one expert gets all... fine.

Potential issue: group_size_m when expert_blocks.numel() < GROUP_M: group_m=1 then, and formula still works? group_m=1, num_pid_in_group = 1*num_pid_n. group_size_m = min(num_pid_m - first_pid_m, 1) = 1. pid_m = first_pid_m + 0. Fine.

tl.dot with BM=16 fine (min 16). FP16 inputs, FP32 accum — allowed tolerance.

expert value: tl.load(EXPERT + pid_m) — pid_m indexes blocks, expert per block. Good.

One subtlety: pid_m within a group: (pid % num_pid_in_group) % group_size_m — but if group is not full (last group), group_size_m smaller; standard vLLM uses this and it works because remaining pids count = group_size_m*num_pid_n.

Numerically: FP16 dot with FP32 accumulation within tolerance 0.01+0.0002|target|. Bounds: |x|,|w| ≤ 0.5, K ≤129 → per-route |proj| ≤ 0.5*0.5*129*... actually sum of 129 terms each ≤0.25 → ≤32.25, times weight sum ≤1 → ≤32.25 < 32.251. Accumulation error in FP32 for tl.dot with FP16 inputs: inputs are FP16 exact, dot uses TF32? No, FP16 inputs → tensor core FP16 with FP32 accumulate, fine. Weight multiply in FP32. Should be within tolerance. Sum in combine FP32. Fine.

topk_weights is FP32 loaded as ROUTE_WEIGHT; tl.load type from pointer — FP32, ok.

Verdict: trust.

## Final response (verbatim)

{"verdict": "trust", "confidence": 0.85, "reason": "The routing preparation (expert-sorted slots with sentinel padding), the projection kernel (masked loads, FP32 tl.dot accumulation, per-route weight application, sentinel-masked stores), and the per-token reduction all faithfully implement the contract; grid/scheduling matches the standard grouped-swathe pattern and edge cases (zero-route experts, partial groups, weight sums ≤ 1) are handled."}

## Verdict

{
  "verdict": "trust",
  "confidence": 0.85,
  "reason": "The routing preparation (expert-sorted slots with sentinel padding), the projection kernel (masked loads, FP32 tl.dot accumulation, per-route weight application, sentinel-masked stores), and the per-token reduction all faithfully implement the contract; grid/scheduling matches the standard grouped-swathe pattern and edge cases (zero-route experts, partial groups, weight sums ≤ 1) are handled."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "trust",
    "confidence": 0.85,
    "reason": "The routing preparation (expert-sorted slots with sentinel padding), the projection kernel (masked loads, FP32 tl.dot accumulation, per-route weight application, sentinel-masked stores), and the per-token reduction all faithfully implement the contract; grid/scheduling matches the standard grouped-swathe pattern and edge cases (zero-route experts, partial groups, weight sums ≤ 1) are handled."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 3260,
    "output_tokens": 979
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 21.98617516690865,
  "kernel_sha256": "9cbead996c8c42d9101713e683ac7f186c5baf50a245620aa2f5ce450e44eaeb",
  "problem_sha256": "349f9a6045f0a827a0cdb93ac18478b22a047584af9af808bd24809e39a1b39a",
  "prompt_variant": "original",
  "estimated_usd": 0.0088716,
  "pricing_snapshot": {
    "input_per_million": 1.4,
    "output_per_million": 4.4,
    "cache_write_multiplier": 1.0,
    "cache_read_multiplier": 1.0,
    "basis": "published list price; cached input conservatively billed as uncached; excludes GPU",
    "checked_at": "2026-10-03",
    "source": "https://fireworks.ai/models/fireworks/glm-5p3"
  },
  "pricing": "dated list-price estimate; not invoice; excludes GPU and unreported HTTP usage"
}
