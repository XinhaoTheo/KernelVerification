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

Compute every state of a two-component affine recurrence, independently for each batch row.

The exact mathematical target is:

    state = initial[row]
    for step in range(N):
        state = matrix[row, step] @ state + bias[row, step]
        output[row, step] = state

The matrix acts on a column vector. Both state components are updated from the same previous state. Every multiply and sum denotes exact integer arithmetic; there is no modulo reduction, saturation, quantization or approximation.

Inputs and complete public domain:
- matrix is an ordinary torch.Tensor with dtype int8 and shape [B,N,2,2]. Every entry is -1, 0 or 1.
- bias is an ordinary torch.Tensor with dtype int16 and shape [B,N,2]. initial is an ordinary torch.Tensor with dtype int16 and shape [B,2]. Every int16 value is legal, including -32768 and 32767.
- 1<=B<=64 and 1<=N<=32. EVERY combination of legal matrix, bias and initial values is allowed. Matrices can be singular or noncommuting. No input distribution, sparsity pattern or seed restriction applies.
- All inputs are dense, strided, contiguous and on the same CUDA device. Contiguous views with nonzero storage offsets are legal. Inputs may share storage if each input satisfies its own requirements.
- Inputs have requires_grad=False, is_inference=False, is_neg=False and is_conj=False. Calls run in ordinary non-inference mode without concurrent external writes. No autograd behavior is required.

Return a fresh int64 tensor with shape [B,N,2] on the input device. Preserve all inputs and their backing storage. Every output element must equal the mathematical target exactly: absolute tolerance=0 and relative tolerance=0. The largest legal state magnitude is less than 2^48, so int64 can represent every required result. Parallelization, internal storage and the order of exact operations are unrestricted if the result satisfies these requirements.

Correctness is one verdict for the whole implementation. An input can demonstrate a defect, but does not get a separate correctness label. validate_inputs() enforces the complete domain and is called by run(). reference() computes the sequential recurrence and also accepts CPU inputs. make_inputs() only supplies examples.

Linear state recurrences and parallel prefix scans are standard computations. This original synthetic kernel uses an 8-step local scan and a tree scan of chunk summaries. The restricted integer domain makes the expected result exact; it is a verification workload, not a production floating-point state-space-model kernel or a claimed historical upstream defect.

Frozen environment: NVIDIA T4, PyTorch2.8.0, Triton3.4.0.


## Kernel implementation under test (kernel.py)

```python
"""Chunked exact affine recurrence with parallel local and block scans.

Original synthetic Triton implementation of a standard linear recurrence.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _compose(la, lb, lc, ld, le, lf, ra, rb, rc, rd, re, rf):
    # A payload (a,b,c,d,e,f) acts as [[a,b],[c,d]] @ state + [e,f].
    # The left interval precedes the right interval.
    a = ra * la + rb * lc
    b = ra * lb + rb * ld
    c = rc * la + rd * lc
    d = rc * lb + rd * ld
    e = ra * le + rb * lf + re
    f = rc * le + rd * lf + rf
    return a, b, c, d, e, f


@triton.jit
def _exclusive_chunks(a, b, c, d, e, f, C: tl.constexpr, LEVELS: tl.constexpr):
    lane = tl.arange(0, C)
    # Build interval totals at the right endpoints of a binary tree.
    for level in tl.static_range(0, LEVELS):
        stride = 1 << level
        take = (lane + 1) % (2 * stride) == 0
        left = tl.maximum(lane - stride, 0)
        la = tl.gather(a, left, 0)
        lb = tl.gather(b, left, 0)
        lc = tl.gather(c, left, 0)
        ld = tl.gather(d, left, 0)
        le = tl.gather(e, left, 0)
        lf = tl.gather(f, left, 0)
        na, nb, nc, nd, ne, nf = _compose(la, lb, lc, ld, le, lf, a, b, c, d, e, f)
        a, b = tl.where(take, na, a), tl.where(take, nb, b)
        c, d = tl.where(take, nc, c), tl.where(take, nd, d)
        e, f = tl.where(take, ne, e), tl.where(take, nf, f)
    a, d = tl.where(lane == C - 1, 1, a), tl.where(lane == C - 1, 1, d)
    b, c = tl.where(lane == C - 1, 0, b), tl.where(lane == C - 1, 0, c)
    e, f = tl.where(lane == C - 1, 0, e), tl.where(lane == C - 1, 0, f)
    # Propagate an exclusive prefix to each subtree.
    for level in tl.static_range(LEVELS - 1, -1, -1):
        stride = 1 << level
        right = (lane // (2 * stride) + 1) * (2 * stride) - 1
        left = right - stride
        pa, pb = tl.gather(a, right, 0), tl.gather(b, right, 0)
        pc, pd = tl.gather(c, right, 0), tl.gather(d, right, 0)
        pe, pf = tl.gather(e, right, 0), tl.gather(f, right, 0)
        la, lb = tl.gather(a, left, 0), tl.gather(b, left, 0)
        lc, ld = tl.gather(c, left, 0), tl.gather(d, left, 0)
        le, lf = tl.gather(e, left, 0), tl.gather(f, left, 0)
        na, nb, nc, nd, ne, nf = _compose(la, lb, lc, ld, le, lf, pa, pb, pc, pd, pe, pf)
        a = tl.where(lane == left, pa, tl.where(lane == right, na, a))
        b = tl.where(lane == left, pb, tl.where(lane == right, nb, b))
        c = tl.where(lane == left, pc, tl.where(lane == right, nc, c))
        d = tl.where(lane == left, pd, tl.where(lane == right, nd, d))
        e = tl.where(lane == left, pe, tl.where(lane == right, ne, e))
        f = tl.where(lane == left, pf, tl.where(lane == right, nf, f))
    return a, b, c, d, e, f


@triton.jit
def _local_scan(A, BIAS, PREFIX, SUMMARY, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    step = chunk * BLOCK + lane
    pos = row * N + step
    a = tl.load(A + pos * 4, step < N, 1).to(tl.int64)
    b = tl.load(A + pos * 4 + 1, step < N, 0).to(tl.int64)
    c = tl.load(A + pos * 4 + 2, step < N, 0).to(tl.int64)
    d = tl.load(A + pos * 4 + 3, step < N, 1).to(tl.int64)
    e = tl.load(BIAS + pos * 2, step < N, 0).to(tl.int64)
    f = tl.load(BIAS + pos * 2 + 1, step < N, 0).to(tl.int64)
    a, b, c, d, e, f = tl.associative_scan((a, b, c, d, e, f), 0, _compose)
    tl.store(PREFIX + pos * 6, a, step < N)
    tl.store(PREFIX + pos * 6 + 1, b, step < N)
    tl.store(PREFIX + pos * 6 + 2, c, step < N)
    tl.store(PREFIX + pos * 6 + 3, d, step < N)
    tl.store(PREFIX + pos * 6 + 4, e, step < N)
    tl.store(PREFIX + pos * 6 + 5, f, step < N)
    summary = (row * C + chunk) * 6 + 0 * lane
    tl.store(SUMMARY + summary, a, lane == BLOCK - 1)
    tl.store(SUMMARY + summary + 1, b, lane == BLOCK - 1)
    tl.store(SUMMARY + summary + 2, c, lane == BLOCK - 1)
    tl.store(SUMMARY + summary + 3, d, lane == BLOCK - 1)
    tl.store(SUMMARY + summary + 4, e, lane == BLOCK - 1)
    tl.store(SUMMARY + summary + 5, f, lane == BLOCK - 1)


@triton.jit
def _block_carries(SUMMARY, INITIAL, CARRY, CHUNKS: tl.constexpr, C: tl.constexpr, LEVELS: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, C)
    pos = (row * C + lane) * 6
    a = tl.load(SUMMARY + pos, lane < CHUNKS, 1)
    b = tl.load(SUMMARY + pos + 1, lane < CHUNKS, 0)
    c = tl.load(SUMMARY + pos + 2, lane < CHUNKS, 0)
    d = tl.load(SUMMARY + pos + 3, lane < CHUNKS, 1)
    e = tl.load(SUMMARY + pos + 4, lane < CHUNKS, 0)
    f = tl.load(SUMMARY + pos + 5, lane < CHUNKS, 0)
    a, b, c, d, e, f = _exclusive_chunks(a, b, c, d, e, f, C, LEVELS)
    x = tl.load(INITIAL + row * 2).to(tl.int64)
    y = tl.load(INITIAL + row * 2 + 1).to(tl.int64)
    tl.store(CARRY + (row * C + lane) * 2, a * x + b * y + e)
    tl.store(CARRY + (row * C + lane) * 2 + 1, c * x + d * y + f)


@triton.jit
def _apply_prefix(PREFIX, CARRY, OUT, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    step = chunk * BLOCK + tl.arange(0, BLOCK)
    pos = row * N + step
    a = tl.load(PREFIX + pos * 6, step < N, 0)
    b = tl.load(PREFIX + pos * 6 + 1, step < N, 0)
    c = tl.load(PREFIX + pos * 6 + 2, step < N, 0)
    d = tl.load(PREFIX + pos * 6 + 3, step < N, 0)
    e = tl.load(PREFIX + pos * 6 + 4, step < N, 0)
    f = tl.load(PREFIX + pos * 6 + 5, step < N, 0)
    x = tl.load(CARRY + (row * C + chunk) * 2)
    y = tl.load(CARRY + (row * C + chunk) * 2 + 1)
    tl.store(OUT + pos * 2, a * x + b * y + e, step < N)
    tl.store(OUT + pos * 2 + 1, c * x + d * y + f, step < N)


def validate_inputs(matrix, bias, initial, require_cuda=True):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    for value in (matrix, bias, initial):
        require(type(value) is torch.Tensor, "ordinary torch.Tensor inputs required")
        require(value.layout == torch.strided and value.is_contiguous(), "contiguous strided inputs required")
        require(not value.requires_grad and not value.is_inference(), "ordinary non-gradient inputs required")
        require(not value.is_neg() and not value.is_conj(), "logical view flags are outside this domain")
        require(not require_cuda or value.is_cuda, "CUDA inputs required by run")
    require(matrix.ndim == 4 and tuple(matrix.shape[-2:]) == (2, 2), "matrix must have shape [B,N,2,2]")
    batch, length = matrix.shape[:2]
    require(1 <= batch <= 64 and 1 <= length <= 32, "B in [1,64] and N in [1,32] required")
    require(tuple(bias.shape) == (batch, length, 2) and tuple(initial.shape) == (batch, 2), "bias [B,N,2] and initial [B,2] required")
    require(matrix.dtype == torch.int8 and bias.dtype == torch.int16 and initial.dtype == torch.int16, "matrix int8; bias and initial int16 required")
    require(matrix.device == bias.device == initial.device, "all inputs must share a device")
    require(bool(((matrix >= -1) & (matrix <= 1)).all().item()), "matrix entries must be -1, 0 or 1")
    return {"valid": True, "batch": batch, "length": length}


def run(matrix, bias, initial):
    validate_inputs(matrix, bias, initial)
    batch, length = matrix.shape[:2]
    block = 8
    chunks = triton.cdiv(length, block)
    padded = triton.next_power_of_2(chunks)
    with torch.cuda.device(matrix.device):
        prefix = torch.empty((batch, length, 6), dtype=torch.int64, device=matrix.device)
        summary = torch.empty((batch, padded, 6), dtype=torch.int64, device=matrix.device)
        carry = torch.empty((batch, padded, 2), dtype=torch.int64, device=matrix.device)
        output = torch.empty((batch, length, 2), dtype=torch.int64, device=matrix.device)
        _local_scan[(batch, chunks)](matrix, bias, prefix, summary, N=length, C=padded, BLOCK=block, num_warps=4)
        _block_carries[(batch,)](summary, initial, carry, CHUNKS=chunks, C=padded, LEVELS=padded.bit_length()-1, num_warps=4)
        _apply_prefix[(batch, chunks)](prefix, carry, output, N=length, C=padded, BLOCK=block, num_warps=4)
    return output


def reference(matrix, bias, initial):
    """Sequential recurrence with int64 arithmetic, independent of the scan."""
    validate_inputs(matrix, bias, initial, require_cuda=False)
    matrices, shifts = matrix.to(torch.int64), bias.to(torch.int64)
    state = initial.to(torch.int64)
    result = []
    for step in range(matrix.shape[1]):
        state = (matrices[:, step] * state[:, None, :]).sum(dim=-1) + shifts[:, step]
        result.append(state)
    return torch.stack(result, dim=1)


def make_inputs(device="cuda", batch=7, length=32, seed=0):
    """Examples only; all values in the public domain are allowed."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randint(-1, 2, (batch, length, 2, 2), generator=generator, dtype=torch.int8)
    bias = torch.randint(-32768, 32768, (batch, length, 2), generator=generator, dtype=torch.int16)
    initial = torch.randint(-32768, 32768, (batch, 2), generator=generator, dtype=torch.int16)
    return matrix.to(device), bias.to(device), initial.to(device)

```

Is this implementation correct?

## Provider reasoning (verbatim)

Check the block scan logic. The affine composition (a,b,c,d,e,f)∘(g,...): left interval precedes right. Applying left then right: new = R∘L, i.e. new matrix = R@L, bias = R@L_bias + R_bias. _compose(l=left, r=right): a = ra*la + rb*lc — right matrix times left matrix. e = ra*le + rb*lf + re — right applied to left's bias plus right's. Correct associative scan with combine of (left, right) → right-after-left? Wait, associative_scan combines elements in order; the combine should produce combined operator such that combined applied to x = right(left(x)). Yes correct.

Local scan: loads with identity defaults for step>=N (identity matrix, zero bias), scan gives inclusive prefix within chunk; stores only step<N. Summary stores last lane (lane==BLOCK-1) which is total over the 8 steps (padded with identity). Good.

_block_carries: exclusive chunks scan. Need to verify _exclusive_chunks. First loop builds tree totals at right endpoints; standard inclusive scan. Then last lane set to identity — making all lanes exclusive of total? Actually after inclusive scan, lane i holds total of 0..i. Setting lane C-1 to identity... hmm, then downward propagation: for each level, right = end of block, left = right-stride. pa = value at right (which after modification is... ). This is a known pattern: exclusive scan via clearing last element to identity and propagating: with inclusive scan values, setting the last element to identity, then downward sweep: at lane==left, set pa (the prefix at right, exclusive of everything after? )... Let me think with small C=2. Inclusive scan: lane0 = s0, lane1 = s0∘s1... wait order: associative_scan with combine(left,right) where left is earlier. lane1 = s1∘s0? Combine(l0, l1) — in _compose l is the first arg (earlier), r second. Actually scan elements combine in order: combine(s0, s1) where s0 earlier = left. Result applied to x: s1(s0(x)). Good.

C=2, LEVELS=1. Inclusive: lane0=s0, lane1=combine(s0,s1)=total. Then identity at lane C-1=1: lane1 = identity. Downward loop level 0: stride=1. right = (lane//2+1)*2-1 → lane0: right=1, left=0. lane1: right=(1+1)*2-1=3, left=2 (out of range). Gather at right: for lane0, pa=a[1]=identity; la=a[0]=s0. na=compose(s0, identity)=s0. lane==left(0) → a=pa=identity. For lane1: pa=a[3] (out of range gather!). tl.gather with out-of-range index... hmm, lane1: right=3, left=2. Gather index 3 out of bounds for C=2. That could be UB. But then tl.where(lane==left(2)? no; lane==right(3)? no) → keeps original a. But the gather itself with out-of-bounds index—Triton gather may return garbage but that's fine since unused? Undefined behavior possibly but usually reads garbage; result still masked out. Hmm, but gather may compute on garbage; values used only in `na` which is discarded for lane1. Fine as long as no fault. Triton gather out-of-bounds typically returns undefined but doesn't fault? Actually it might fault. Hmm.

Also for lane0 after downward: lane0 = pa = a[1] = identity = exclusive prefix for chunk 0. Correct. lane1 = na = compose(s0, identity) = s0 = exclusive prefix of chunk1. Correct.

Larger C: exclusive prefix at lane i should be compose of summaries 0..i-1 applied... Actually exclusive scan: lane i holds combine of s_0..s_{i-1} with identity at lane 0. The standard algorithm: inclusive scan, then replace last element with identity, then downward propagate: at each node, left child gets parent's exclusive value, right child gets parent's exclusive ∘ left child's total. Looks correct (Hillis-Steele then down-sweep). The out-of-range gathers: right computed as (lane//(2*stride)+1)*(2*stride)-1 for lanes near end could exceed C-1 when C not multiple? C is padded power of 2, and lanes 0..C-1. For lane in last block of size 2*stride: right = C-1 ≤ C-1 fine. Actually (lane//(2s)+1)*2s-1: max lane = C-1, lane//(2s) ≤ (C-1)/(2s), +1 times 2s -1 ≤ ... e.g. C=4, stride=1, lane=3: (3//2+1)*2-1=3. OK in range. In my C=2 example, lane1, stride=1: (1//2+1)*2-1 = 1. I miscalculated: lane=1, 2*stride=2, 1//2=0, (0+1)*2-1=1. Right=1, left=0. OK, in range. Good, no OOB.

So exclusive scan gives lane i = combine(s_0,...,s_{i-1}) (exclusive). Then carry = exclusive_total applied to initial. Correct: chunk c's prefix state = (composition of steps 0..c*8-1) applied to initial. Wait exclusive composition order: combine(s0,s1) applies s1 after s0. Exclusive at lane i = s_{i-1}∘...∘s_0 applied → yes.

But careful: LEVELS = padded.bit_length()-1. For padded=1 (N≤8, chunks=1): LEVELS=0. _exclusive_chunks with LEVELS=0: no first loop, then lane C-1=0 set to identity. Downward loop from -1 to 0: static_range(-1,-1,-1) empty? tl.static_range(LEVELS-1,-1,-1) = static_range(-1,-1,-1) — empty. Returns identity. carry = identity applied to initial = initial. Correct.

But note in _exclusive_chunks, out-of-bounds summary lanes: _block_carries loads summary with lane < CHUNKS, padding with identity. Good.

_apply_prefix: loads prefix (inclusive within-chunk scan), carry for chunk; out = within-chunk inclusive prefix applied to carry state. Correct.

Also check first loop take condition: (lane+1) % (2*stride)==0 — lanes at right endpoints of blocks. compose(left = lane-stride, right = lane). Correct Hillis-Steele.

Concern: _compose uses int64 throughout (loaded as int64). Stores fine. tl.associative_scan with tuple and custom combine — supported in Triton 3.4.

One subtlety: in _local_scan, gather/scan with BLOCK=8 fixed. associative_scan on axis 0 over lane of size 8. Fine.

Also dtype: _block_carries loads summary as int64 (stored int64), .to(tl.int64) not needed but ok. INITIAL .to(tl.int64) — initial is int16 pointer; loads int16 then cast. Fine.

Padded C: chunks up to 4 (N=32, block 8 → chunks=4, padded=4, LEVELS=2). Fine.

Also summary stored via `summary = (row*C + chunk)*6 + 0*lane` — a scalar pointer plus mask lane==BLOCK-1; store scalar with mask of vector... storing `a` (vector) with scalar pointer and mask? Hmm: `tl.store(SUMMARY + summary, a, lane == BLOCK - 1)` — pointer scalar, value vector, mask vector. Triton broadcasts pointer to vector shape. OK.

Verdict: trust.

## Final response (verbatim)

{"verdict": "trust", "confidence": 0.82, "reason": "The affine composition operator is associative and correctly ordered (right interval applied after left), the local associative_scan produces within-chunk inclusive prefixes with identity padding, the Hillis-Steele up-sweep/down-sweep _exclusive_chunks yields correct exclusive chunk carries, and _apply_prefix composes them exactly in int64, so results match the reference for all legal inputs including edge cases N<=8 (LEVELS=0) and non-power-of-two N."}

## Verdict

{
  "verdict": "trust",
  "confidence": 0.82,
  "reason": "The affine composition operator is associative and correctly ordered (right interval applied after left), the local associative_scan produces within-chunk inclusive prefixes with identity padding, the Hillis-Steele up-sweep/down-sweep _exclusive_chunks yields correct exclusive chunk carries, and _apply_prefix composes them exactly in int64, so results match the reference for all legal inputs including edge cases N<=8 (LEVELS=0) and non-power-of-two N."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "trust",
    "confidence": 0.82,
    "reason": "The affine composition operator is associative and correctly ordered (right interval applied after left), the local associative_scan produces within-chunk inclusive prefixes with identity padding, the Hillis-Steele up-sweep/down-sweep _exclusive_chunks yields correct exclusive chunk carries, and _apply_prefix composes them exactly in int64, so results match the reference for all legal inputs including edge cases N<=8 (LEVELS=0) and non-power-of-two N."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 3962,
    "output_tokens": 1862
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 30.099181666970253,
  "kernel_sha256": "5d018e4f9387e31827969915baef453929000fd931a8baacf01455b87e3e0d84",
  "problem_sha256": "f15d89e08ca39b4d28fd813816499a08ec72700878f1305518d2db49a218561a",
  "prompt_variant": "original",
  "estimated_usd": 0.013739600000000001,
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
