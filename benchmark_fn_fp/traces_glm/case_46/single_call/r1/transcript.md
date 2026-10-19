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
    pid_n = (pid % num_pid_in_group) // GROUP_M
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

Let me analyze the kernel. The MoE kernel: sorted_ids per expert blocks, EXPERT array indexed by pid_m — but pid_m is block index across all expert blocks, and EXPERT has one entry per block, so tl.load(EXPERT + pid_m) is correct since pid_m indexes blocks 0..num_blocks-1. Good.

Wait, grid = expert_blocks.numel() * cdiv(n,32). The pid mapping: pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m. pid_n = (pid % num_pid_in_group) // GROUP_M. Hmm, standard vLLM swizzle uses group_size_m not GROUP_M for pid_n. In vLLM: pid_n = (pid % num_pid_in_group) // group_size_m. Using GROUP_M instead of group_size_m: when last group has fewer rows, pid_n could exceed num_pid_n? Let's check: pid ranges over total = num_pid_m*num_pid_n grid size. num_pid_in_group = GROUP_M*num_pid_n. In last group, pid offsets 0..(group_size_m*num_pid_n - 1). With GROUP_M: pid_n = offset // GROUP_M can reach up to (group_size_m*num_pid_n-1)//GROUP_M which may be >= num_pid_n if group_size_m < GROUP_M. E.g. GROUP_M=4, num_pid_n=5, group_size_m=2: offsets up to 9, pid_n up to 9//4=2 < 5 fine. Actually max pid_n = (group_size_m*num_pid_n-1)//GROUP_M < num_pid_n when group_size_m <= GROUP_M? group_size_m*num_pid_n/GROUP_M <= num_pid_n iff group_size_m <= GROUP_M. Yes so pid_n < num_pid_n. Good. But pid_m: offset % num_pid_in_group could be >= group_size_m*num_pid_n? No — grid size limits offsets within group to group_size_m*num_pid_n... actually grid total = num_pid_m*num_pid_n. Last group offsets range 0..(remaining*num_pid_n - 1) where remaining = group_size_m. So offset % num_pid_in_group = offset (since offset < num_pid_in_group? offset max = group_size_m*num_pid_n-1 <= GROUP_M*num_pid_n -1, yes). pid_m = first + offset % group_size_m. Coverage: each (pid_m within group, pid_n) pair? offset spans 0..G*N-1; pid_n=offset//GROUP_M, pid_m=offset%group_size_m. Are all (m,n) combos covered? offset o: for o in 0..G*N-1, pairs (o%g, o//GROUP_M). Since g<=GROUP_M, some pairs may be missed and some duplicated? Let's check: g=2, GROUP_M=4, N(num_pid_n)=5, offsets 0..9. o//4: 0,0,0,0,1,1,1,1,2,2. o%2: 0,1,0,1,0,1,0,1,0,1. pid_n=2 has pid_m {0,1}. pid_n=0: {0,1}, pid_n=1: {0,1}. All 10 pairs distinct? o=0→(0,0), 1→(1,0), 2→(0,0)?? o=2: pid_n=0, pid_m=2%2=0. Duplicate (0,0)! And pid_n=3 missing. So tile (0, n-col 0) computed twice and tile pid_n=3 never computed → missing output tiles. That's the standard swizzle bug — vLLM uses group_size_m for pid_n division. Here using GROUP_M causes duplicates and missed tiles when last group is partial.

Wait but check standard vLLM: pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m; pid_n = (pid % num_pid_in_group) // group_size_m. Yes. Here GROUP_M is used instead. Bug confirmed... but wait, group_m = 4 if expert_blocks.numel() >= 4 else 1. When does the last group become partial? num_pid_m = EM blocks = total expert blocks. If expert_blocks.numel() not multiple of 4, last group partial → missed tiles. E.g. 5 blocks: groups {0-3},{4}. Group 1 has group_size_m=1, GROUP_M=4. offsets 0..num_pid_n-1. pid_n = offset//4 = 0 always! Only pid_n=0 computed; pid_n 1.. missed. So yes real defect.

Wait—but pid_m computed = first + offset % group_size_m = first + 0 = correct block. pid_n = offset//4 = 0 for offset<num_pid_n. So all N tiles except n-block 0 are never computed → routed zeros → wrong output. Reject.

Also check: does last group partial occur commonly? Yes whenever total blocks % 4 != 0 with GROUP_M=4. Definite reject.

Also check pid_n indexing generally: even in a full group, GROUP_M == group_size_m so fine. So bug only in partial groups, which occur for many expert counts (e.g. 5 blocks). Definitely a real defect.

## Final response (verbatim)

{"verdict": "reject", "confidence": 0.9, "reason": "The swizzle in _expert_projection computes pid_n with '// GROUP_M' instead of '// group_size_m' (the vLLM original). When the last M-group is partial (total expert blocks not a multiple of GROUP_M=4), offsets in that group yield pid_n values that skip whole N-tiles (e.g. with a single trailing block, pid_n = offset//4 = 0 for all offsets, so only the first N-block is computed), leaving output columns wrong/zero."}

## Verdict

{
  "verdict": "reject",
  "confidence": 0.9,
  "reason": "The swizzle in _expert_projection computes pid_n with '// GROUP_M' instead of '// group_size_m' (the vLLM original). When the last M-group is partial (total expert blocks not a multiple of GROUP_M=4), offsets in that group yield pid_n values that skip whole N-tiles (e.g. with a single trailing block, pid_n = offset//4 = 0 for all offsets, so only the first N-block is computed), leaving output columns wrong/zero."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "reject",
    "confidence": 0.9,
    "reason": "The swizzle in _expert_projection computes pid_n with '// GROUP_M' instead of '// group_size_m' (the vLLM original). When the last M-group is partial (total expert blocks not a multiple of GROUP_M=4), offsets in that group yield pid_n values that skip whole N-tiles (e.g. with a single trailing block, pid_n = offset//4 = 0 for all offsets, so only the first N-block is computed), leaving output columns wrong/zero."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 3259,
    "output_tokens": 1270
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 22.134610916953534,
  "kernel_sha256": "c9d60e071674accd285a9c0f8114336b1632641499b9147bbe13cb58971f3dfb",
  "problem_sha256": "349f9a6045f0a827a0cdb93ac18478b22a047584af9af808bd24809e39a1b39a",
  "prompt_variant": "original",
  "estimated_usd": 0.0101506,
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
