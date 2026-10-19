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

Compute a matrix product followed by a column-bias epilogue:
  target[i,j] = sum(A[i,k] * B[k,j], k=0..K-1) + bias[j].
The target is real arithmetic on the values of the provided tensors. This is
forward linear-layer evaluation, not a definition of rounded intermediate
results. An implementation may partition the inner dimension (split-K).

Inputs:
- A is FP16 [M,K] and B is FP16 [K,N], with all magnitudes <= 0.5.
- bias is FP32 [N], with all magnitudes <= 1.
- 1 <= M <= 64, 1 <= N <= 64, 1 <= K <= 512.
- split_k is a Python int in {1,2,4,8}; it is only a partition parameter and
  never changes the required result. K need not be divisible by split_k or 32.

All values are finite; zeros and subnormal values are permitted. All tensors
must be ordinary torch.Tensor objects (not subclasses), dense, strided,
contiguous, on the same CUDA device, with disjoint occupied storage ranges.
They have requires_grad=False, is_inference=False, is_neg=False, is_conj=False.
This interface consumes materialized numerical buffers and does not provide
autograd. Calls use ordinary non-inference execution mode with no concurrent
external writes. Nonzero storage offsets are permitted when these conditions
hold. Preserve every input.

Return a new FP32 CUDA tensor [M,N] on the input device. Each output must be
finite and satisfy abs(actual-target) <= 0.01 + 0.001 * abs(target).
No intermediate rounding behavior is part of the target. The bounded domain
and tolerance accommodate ordinary FP32 accumulation; the largest exact
unscaled partial-sum magnitude is at most 128, well within finite FP16 range.

validate_inputs() is public and is actually invoked by run(). reference()
independently uses a full FP64 matrix multiplication and accepts CPU inputs.
make_inputs() gives one example only; correctness covers every legal shape,
value and split count, not the generator's seeds. Frozen environment: NVIDIA
T4, PyTorch 2.8.0, Triton 3.4.0.


## Kernel implementation under test (kernel.py)

```python
"""Standalone split-K matrix multiplication with a bias epilogue.

Inspired by Triton's split-K scratch / final-reduction design.
Copyright 2018-2020 Philippe Tillet; Copyright 2020-2022 OpenAI.
MIT license; see LICENSE. This is a bounded T4 adaptation, not upstream code.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partials(A, B, WORK, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              TILES: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    lane = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    for tile in range(TILES):
        inner = (split * TILES + tile) * BK + lane
        left = tl.load(A + row[:, None] * K + inner[None, :],
                       mask=(row[:, None] < M) & (inner[None, :] < K), other=0.)
        right = tl.load(B + inner[:, None] * N + col[None, :],
                        mask=(inner[:, None] < K) & (col[None, :] < N), other=0.)
        accum = tl.dot(left, right, accum)
    offset = split * M * N + row[:, None] * N + col[None, :]
    tl.store(WORK + offset, accum, mask=(row[:, None] < M) & (col[None, :] < N))


@triton.jit
def _finish(WORK, BIAS, OUT, M: tl.constexpr, N: tl.constexpr,
            SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.zeros((BLOCK,), tl.float32)
    for split in range(SPLITS):
        partial = tl.load(WORK + split * M * N + index, mask=index < M * N, other=0.)
        value += partial.to(tl.float32)
    bias = tl.load(BIAS + index % N, mask=index < M * N, other=0.)
    tl.store(OUT + index, value + bias, mask=index < M * N)


def validate_inputs(a, b, bias, split_k, require_cuda=True):
    """Check the complete public domain; reference checks can also run on CPU."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    values = (a, b, bias)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    for tensor in values:
        require(type(tensor) is torch.Tensor, "standard torch.Tensor required")
        require(tensor.layout == torch.strided and tensor.is_contiguous(), "contiguous strided tensor required")
        require(not tensor.requires_grad and not tensor.is_inference(), "ordinary non-gradient tensors required")
        require(not tensor.is_neg() and not tensor.is_conj(), "materialized logical values required")
    require(all(t.device == a.device for t in values), "all tensors must share a device")
    require(not require_cuda or a.is_cuda, "run requires CUDA")
    require(a.ndim == b.ndim == 2 and bias.ndim == 1, "expected two matrices and a bias vector")
    m, k = a.shape
    kb, n = b.shape
    require(k == kb and bias.shape == (n,), "matrix and bias dimensions disagree")
    require(1 <= m <= 64 and 1 <= n <= 64 and 1 <= k <= 512, "shape outside supported domain")
    require(a.dtype == b.dtype == torch.float16 and bias.dtype == torch.float32, "A/B must be FP16 and bias FP32")
    require(type(split_k) is int and split_k in (1, 2, 4, 8), "split_k must be 1, 2, 4 or 8")
    for tensor, bound in ((a, .5), (b, .5), (bias, 1.)):
        require(bool(torch.isfinite(tensor).all().item()), "finite values required")
        require(bool((tensor.abs() <= bound).all().item()), "input magnitude outside domain")
    intervals = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in values]
    require(all(a1 <= b0 or b1 <= a0 for i, (a0, a1) in enumerate(intervals)
                for b0, b1 in intervals[i + 1:]), "input storage ranges must be disjoint")
    return {"valid": True, "m": m, "n": n, "k": k, "split_k": split_k}


def run(a, b, bias, split_k=4):
    """Return a fresh FP32 CUDA result; input buffers are preserved."""
    validate_inputs(a, b, bias, split_k)
    m, k = a.shape
    n = b.shape[1]
    tiles = triton.cdiv(triton.cdiv(k, 32), split_k)
    work_dtype = a.dtype if split_k > 1 else torch.float32
    work = torch.empty((split_k, m, n), dtype=work_dtype, device=a.device)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    with torch.cuda.device(a.device):
        _partials[(triton.cdiv(m, 16), triton.cdiv(n, 16), split_k)](
            a, b, work, M=m, N=n, K=k, TILES=tiles, BM=16, BN=16, BK=32,
            num_warps=4, num_stages=2)
        _finish[(triton.cdiv(m * n, 256),)](
            work, bias, out, M=m, N=n, SPLITS=split_k, BLOCK=256, num_warps=4)
    return out


def reference(a, b, bias, split_k=4):
    """FP64 full matrix product; independent of split count and scratch layout."""
    validate_inputs(a, b, bias, split_k, require_cuda=False)
    return torch.matmul(a.double(), b.double()) + bias.double()[None, :]


def make_inputs(device="cuda", m=17, n=33, k=257, split_k=4, seed=0):
    """A convenience example, not a restriction on the legal input values."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = (torch.rand((m, k), generator=generator) - .5).half()
    b = (torch.rand((k, n), generator=generator) - .5).half()
    bias = (2 * torch.rand((n,), generator=generator) - 1).float()
    return a.to(device), b.to(device), bias.to(device), split_k

```

Is this implementation correct?

## Provider reasoning (verbatim)

Key issue: work dtype is FP16 when split_k > 1 — partial sums stored in FP16, then summed. Is that within tolerance? Max exact partial sum: 0.5*0.5*512 = 128, split-K partials smaller. FP16 rounding of partials... partial magnitude up to 128, FP16 precision ~2^-8 relative → error up to ~0.5 per split? Actually fp16 near 128 has ulp = 0.0625. Sum of 8 partials each rounded → up to ~0.25 error. Tolerance: 0.01 + 0.001*|target|. Target could be small (near zero) with error up to 0.25 → violation. Hmm, but actual errors: each partial up to |0.5·0.5·K/splits|·... With K=512, split=8, each partial magnitude ≤ 8, fp16 ulp at 8 is 0.0078... but intermediate accumulation in tl.dot is fp32, only final store rounded. So per-split rounding error ≤ ulp/2 ≈ 0.004. Sum of 8 → ≤0.03? Actually sum of errors ≤ 8*0.004 = 0.03 > 0.01. But magnitudes rarely max. Still contract says worst case must satisfy. Also: fp16 range fine (max 128 < 65504). Also could error exceed tolerance when target small? Partial could be 8 with target sum ≈ 0; error ~0.004 each, total ~0.03 > 0.01. This is a real defect risk. Standard Triton split-k examples store in input dtype... but contract explicitly says "No intermediate rounding behavior is part of the target" and tolerance accommodates ordinary FP32 accumulation. Storing partials in FP16 is a genuine defect (reject).

Also check bias indexing: index % N works for contiguous row-major. Fine.

tiles computation: cdiv(cdiv(k,32), split_k)*BK*split_k may not cover K fully? tiles = cdiv(cdiv(k,32), split). Total inner coverage = split*tiles*32 ≥ cdiv(k,32)*32 ≥ k. Yes covered.

So reject due to FP16 partial storage.

## Final response (verbatim)

{"verdict": "reject", "confidence": 0.8, "reason": "When split_k > 1 the partial sums are stored in an FP16 workspace (work_dtype = a.dtype) and then accumulated, introducing intermediate FP16 rounding that the contract explicitly excludes ('no intermediate rounding behavior is part of the target'); worst-case per-split magnitudes near 128 have FP16 ulp up to 0.0625, so accumulated rounding error can exceed the 0.01 + 0.001*|target| tolerance (e.g., near-cancelling partials with small targets). Everything else (coverage of K, masking, bias indexing, disjointness check) looks sound."}

## Verdict

{
  "verdict": "reject",
  "confidence": 0.8,
  "reason": "When split_k > 1 the partial sums are stored in an FP16 workspace (work_dtype = a.dtype) and then accumulated, introducing intermediate FP16 rounding that the contract explicitly excludes ('no intermediate rounding behavior is part of the target'); worst-case per-split magnitudes near 128 have FP16 ulp up to 0.0625, so accumulated rounding error can exceed the 0.01 + 0.001*|target| tolerance (e.g., near-cancelling partials with small targets). Everything else (coverage of K, masking, bias indexing, disjointness check) looks sound."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "reject",
    "confidence": 0.8,
    "reason": "When split_k > 1 the partial sums are stored in an FP16 workspace (work_dtype = a.dtype) and then accumulated, introducing intermediate FP16 rounding that the contract explicitly excludes ('no intermediate rounding behavior is part of the target'); worst-case per-split magnitudes near 128 have FP16 ulp up to 0.0625, so accumulated rounding error can exceed the 0.01 + 0.001*|target| tolerance (e.g., near-cancelling partials with small targets). Everything else (coverage of K, masking, bias indexing, disjointness check) looks sound."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 2302,
    "output_tokens": 636
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 11.191122042015195,
  "kernel_sha256": "a7a8a29ff59ea749b05809a4bf7497e5a467fee13f3177a2f53a87b88427e752",
  "problem_sha256": "7c7bb2039fa7aec3d4152809318892c124991507c507e29c960f8052785e265d",
  "prompt_variant": "original",
  "estimated_usd": 0.0060212,
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
