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

Compute a packed collection of independent state-space recurrences. This is
forward state propagation, with one decay shared by all state coordinates of
one head. The implementation may partition computation into chunks.

Inputs:
- u: FP32 [B,L,H,D], with |u| <= 1.
- decay: FP32 [B,L,H], every value in [0.5,0.96875].
- seq_idx: int32 [B,L], identifying contiguous independent sequences.
  Each row starts at label zero, labels are nondecreasing, and 0 <= label < L.
  A larger label starts a new sequence; labels need not be consecutive.
- initial_state: FP32 [B,H,D], with magnitude <= 1, used only for the first
  sequence in each packed batch row. Subsequent sequences start from zero.
- chunk_size: Python int, one of 16, 32, 64; it is an implementation partition
  parameter and does not change the required mathematical result.
- 1 <= B <= 2, 1 <= L <= 257, 1 <= H <= 4, 1 <= D <= 96.

All tensors must be ordinary torch.Tensor objects (not subclasses), dense,
strided, contiguous, on one CUDA device, with disjoint storage ranges. Floating
values must be finite. Inputs must have requires_grad=False, is_inference=False,
is_neg=False and is_conj=False. This interface concerns materialized numerical
buffers and does not provide autograd. Calls run in ordinary non-inference
execution mode without concurrent external writes to the inputs.

Define a separate state vector for each batch b and head h. At token t:
  prior = initial_state[b,h,:]                 if t == 0
          output[b,t-1,h,:]                   if seq_idx[b,t] == seq_idx[b,t-1]
          0                                  otherwise
  output[b,t,h,:] = decay[b,t,h] * prior + u[b,t,h,:]
These equations specify real arithmetic on the provided FP32 input values.
Return (output, final_state), both FP32 CUDA tensors, with shapes [B,L,H,D]
and [B,H,D], where final_state equals the last required state of each batch/head.
Preserve every input. All output elements must be finite, and each must satisfy:
  abs(actual - target) <= 0.002 + 0.0001 * abs(target).
The stable decay and input bounds imply the exact state's magnitude is at most
32; the tolerance allows ordinary FP32 evaluation and regrouping.

The public validate_inputs() checks the input domain and is called by run().
The public reference() independently evaluates the serial recurrence in FP64
and also accepts CPU inputs. make_inputs() supplies only an example; any legal
values, shapes and sequence lengths may be tested. Correctness covers the full
stated domain. Frozen environment: NVIDIA T4, PyTorch 2.8.0, Triton 3.4.0.


## Kernel implementation under test (kernel.py)

```python
"""Standalone packed state-space recurrence, adapted from Mamba state passing.

Copyright (c) 2024, Tri Dao, Albert Gu. Apache-2.0; see LICENSE.
2026 adaptation: bounded FP32 recurrence, explicit chunk-local and output stages,
fixed launch configuration, standalone validation and reference.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _local_chunks(U, A, IDS, LOCAL, PREFIX, SUMMARY, SCALE,
                  L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  C: tl.constexpr, K: tl.constexpr, BD: tl.constexpr):
    chunk = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    cols = tl.arange(0, BD)
    state = tl.zeros((BD,), tl.float32)
    prefix = tl.full((), 1.0, tl.float32)
    begin = chunk * K
    previous = tl.load(IDS + batch * L + begin)
    for j in range(K):
        token = begin + j
        valid = token < L
        current = tl.load(IDS + batch * L + token, mask=valid, other=-1)
        alpha = tl.load(A + (batch * L + token) * H + head, mask=valid, other=1.0)
        offset = ((batch * L + token) * H + head) * D + cols
        value = tl.load(U + offset, mask=valid & (cols < D), other=0.0)
        prior = tl.where(current == previous, state, 0.0)
        updated = alpha * prior + value
        state = tl.where(valid, updated, state)
        prefix *= alpha
        tl.store(LOCAL + offset, state, mask=valid & (cols < D))
        tl.store(PREFIX + (batch * L + token) * H + head, prefix, mask=valid)
        previous = current
    summary_offset = ((batch * C + chunk) * H + head) * D + cols
    tl.store(SUMMARY + summary_offset, state, mask=cols < D)
    tl.store(SCALE + (batch * C + chunk) * H + head, prefix)


@triton.jit
def _pass_states(SUMMARY, SCALE, IDS, INITIAL, CARRY, FINAL,
                 L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 C: tl.constexpr, K: tl.constexpr, BD: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BD)
    state = tl.load(INITIAL + (batch * H + head) * D + cols,
                    mask=cols < D, other=0.0)
    sequence = tl.full((), 0, tl.int32)
    for chunk in range(C):
        offset = ((batch * C + chunk) * H + head) * D + cols
        tl.store(CARRY + offset, state, mask=cols < D)
        new_state = tl.load(SUMMARY + offset, mask=cols < D, other=0.0)
        factor = tl.load(SCALE + (batch * C + chunk) * H + head)
        representative = chunk * K
        next_sequence = tl.load(IDS + batch * L + representative)
        factor = tl.where(next_sequence == sequence, factor, 0.0)
        state = factor * state + new_state
        sequence = next_sequence
    tl.store(FINAL + (batch * H + head) * D + cols, state, mask=cols < D)


@triton.jit
def _combine(LOCAL, PREFIX, IDS, CARRY, OUT,
             L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
             C: tl.constexpr, K: tl.constexpr, BD: tl.constexpr):
    chunk = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    rows = chunk * K + tl.arange(0, K)
    cols = tl.arange(0, BD)
    previous = tl.load(IDS + batch * L + chunk * K - 1, mask=chunk > 0, other=0)
    current = tl.load(IDS + batch * L + rows, mask=rows < L, other=-1)
    prefix = tl.load(PREFIX + (batch * L + rows) * H + head, mask=rows < L, other=0.0)
    prefix = tl.where(current == previous, prefix, 0.0)
    incoming = tl.load(CARRY + ((batch * C + chunk) * H + head) * D + cols,
                       mask=cols < D, other=0.0)
    offset = ((batch * L + rows[:, None]) * H + head) * D + cols[None, :]
    valid = (rows[:, None] < L) & (cols[None, :] < D)
    local = tl.load(LOCAL + offset, mask=valid, other=0.0)
    result = local + prefix[:, None] * incoming[None, :]
    tl.store(OUT + offset, result, mask=valid)


def validate_inputs(u, decay, seq_idx, initial_state, chunk_size, require_cuda=True):
    """Raise ValueError for inputs outside the public domain; usable on CPU too."""
    values = (u, decay, seq_idx, initial_state)
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(not torch.is_inference_mode_enabled(), "ordinary execution mode required")
    for tensor in values:
        require(type(tensor) is torch.Tensor, "standard torch.Tensor required")
        require(tensor.layout == torch.strided and tensor.is_contiguous(), "contiguous strided tensor required")
        require(not tensor.requires_grad and not tensor.is_inference(), "materialized non-gradient ordinary tensors required")
        require(not tensor.is_neg() and not tensor.is_conj(), "logical view flags are outside this domain")
    require(all(t.device == u.device for t in values), "all inputs must share a device")
    require(not require_cuda or u.is_cuda, "CUDA tensors required by run")
    require(u.ndim == 4, "u must have shape [B,L,H,D]")
    batch, length, heads, width = u.shape
    require(1 <= batch <= 2 and 1 <= length <= 257 and 1 <= heads <= 4 and 1 <= width <= 96, "shape outside supported domain")
    require(type(chunk_size) is int and chunk_size in (16, 32, 64), "chunk_size must be 16, 32 or 64")
    require(u.dtype == decay.dtype == initial_state.dtype == torch.float32, "numeric inputs must be FP32")
    require(seq_idx.dtype == torch.int32, "seq_idx must be int32")
    require(decay.shape == (batch, length, heads), "decay shape mismatch")
    require(seq_idx.shape == (batch, length), "seq_idx shape mismatch")
    require(initial_state.shape == (batch, heads, width), "initial_state shape mismatch")
    for tensor in (u, decay, initial_state):
        require(bool(torch.isfinite(tensor).all().item()), "numeric values must be finite")
    require(bool((u.abs() <= 1).all().item()), "u magnitude exceeds one")
    require(bool((initial_state.abs() <= 1).all().item()), "initial state magnitude exceeds one")
    require(bool(((decay >= .5) & (decay <= .96875)).all().item()), "decay outside [0.5,0.96875]")
    require(bool((seq_idx[:, 0] == 0).all().item()), "each packed row must begin at sequence zero")
    require(bool(((seq_idx >= 0) & (seq_idx < length)).all().item()), "sequence labels outside range")
    require(bool((seq_idx[:, 1:] >= seq_idx[:, :-1]).all().item()), "sequence labels must be nondecreasing")
    intervals = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in values]
    require(all(a1 <= b0 or b1 <= a0 for i, (a0, a1) in enumerate(intervals)
                for b0, b1 in intervals[i + 1:]), "input storage ranges must be disjoint")
    return {"valid": True, "batch": batch, "length": length,
            "heads": heads, "width": width, "chunk_size": chunk_size}


def run(u, decay, seq_idx, initial_state, chunk_size):
    """Return per-token states and the last state; preserve all inputs."""
    validate_inputs(u, decay, seq_idx, initial_state, chunk_size)
    batch, length, heads, width = u.shape
    chunks = triton.cdiv(length, chunk_size)
    local = torch.empty_like(u)
    prefix = torch.empty_like(decay)
    summary = torch.empty((batch, chunks, heads, width), device=u.device, dtype=torch.float32)
    scale = torch.empty((batch, chunks, heads), device=u.device, dtype=torch.float32)
    carry = torch.empty_like(summary)
    output = torch.empty_like(u)
    final = torch.empty_like(initial_state)
    meta = dict(L=length, H=heads, D=width, C=chunks, K=chunk_size,
                BD=triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False)
    with torch.cuda.device(u.device):
        _local_chunks[(chunks, batch, heads)](u, decay, seq_idx, local, prefix, summary, scale, **meta)
        _pass_states[(batch, heads)](summary, scale, seq_idx, initial_state, carry, final, **meta)
        _combine[(chunks, batch, heads)](local, prefix, seq_idx, carry, output, **meta)
    return output, final


def reference(u, decay, seq_idx, initial_state, chunk_size):
    """FP64 serial recurrence, independent of the implementation's chunk layout."""
    validate_inputs(u, decay, seq_idx, initial_state, chunk_size, require_cuda=False)
    u64, a64 = u.double(), decay.double()
    state = initial_state.double().clone()
    result = []
    for token in range(u.shape[1]):
        if token:
            keep = (seq_idx[:, token] == seq_idx[:, token - 1])[:, None, None]
            state = torch.where(keep, state, torch.zeros_like(state))
        state = a64[:, token, :, None] * state + u64[:, token]
        result.append(state.clone())
    return torch.stack(result, dim=1), state


def make_inputs(device="cuda", length=97, chunk_size=32, seed=0):
    """Convenience inputs; legal inputs are not restricted to this generator."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    u = (2 * torch.rand((1, length, 2, 33), generator=generator) - 1).float()
    decay = (.90 + .06 * torch.rand((1, length, 2), generator=generator)).float()
    initial = (2 * torch.rand((1, 2, 33), generator=generator) - 1).float()
    seq_idx = torch.zeros((1, length), dtype=torch.int32)
    if length >= 3:
        seq_idx[:, length // 3:] += 1
        seq_idx[:, 2 * length // 3:] += 1
    return u.to(device), decay.to(device), seq_idx.to(device), initial.to(device), chunk_size

```

Is this implementation correct?

## Provider reasoning (verbatim)

Analyze the chunked scan. Local: within a chunk, prior = state if same seq label else 0. Stores LOCAL state per token (chunk-local, ignoring initial carry). PREFIX is cumulative product of alpha within chunk (including invalid tokens with alpha=1.0? valid false, alpha loaded as other=1.0, prefix *= 1.0 — fine since token invalid beyond L).

Pass states: carry per chunk = state at chunk start. For chunk 0 carry = initial. factor = chunk's prefix product; but if chunk's first token starts a new sequence (seq label differs from previous chunk's label at last valid token?), factor should be 0 — carry shouldn't propagate. Logic: sequence = 0; for each chunk: representative = chunk*K (first token of chunk); next_sequence = label there. If next_sequence == sequence, factor kept, else 0 — meaning if the first token of chunk starts a new sequence, the carry is dropped. But what if a sequence boundary occurs mid-chunk? Then carry for that chunk was already stored (before the update); the carry is the state at chunk start, correct. And the factor=0 handles when the chunk starts a new sequence... but wait: what if the sequence starts mid-chunk? carry stored for the chunk is the state at chunk start, which is still within the previous sequence. Then in _combine, prefix at tokens in the new sequence: prefix = cumulative product from chunk start to that token; they set prefix=0 where current != previous (token's label differs from previous token's label). Actually in _combine, prefix is multiplied by incoming only where current == previous? They compute prefix then zero it where current != previous. But what about tokens after a mid-chunk boundary within the same sequence? E.g., tokens t0,t1 in seq A, t2,t3 in seq B (within one chunk). For token t3, output = local[t3] + prefix[t3]*incoming. local[t3] was computed with prior = 0 at t2 (label change), so local[t3] is the correct state for sequence B starting from zero. prefix[t3]*incoming would incorrectly add prefix[t3]*carry! prefix[t3] is product of alphas over t0..t3, and incoming is chunk-start carry — but sequence B started at t2 from zero, so adding prefix[t3]*incoming is wrong. Hmm — wait, prefix should be zeroed at sequence start, but the carry contribution after a mid-chunk restart should be zero. In _combine, prefix zeroed only at the token where current != previous (t2). For t3, current == previous (both B), prefix nonzero, adds carry incorrectly.

Hmm, but is this wrong? The recurrence for sequence B: state at t3 = a3*(a2*0 + u2) + u3, which is local[t3] (local recurrence within chunk correctly handles boundary). Adding prefix[t3]*incoming where incoming = state at chunk start (end of seq A) is wrong. So mid-chunk sequence boundaries produce incorrect results?

Wait — but prefix in _local_chunks: prefix is the cumulative product of alphas from chunk start, not reset at sequence boundary. In _combine, prefix zeroed only where the token itself is the first of a new sequence. Tokens after that in the new sequence keep nonzero prefix → they get carry contributions from the previous sequence. That's a bug... unless pass_states sets factor=0 when chunk starts a new sequence — but the boundary is mid-chunk, so chunk's first token label equals previous chunk's end label, factor kept.

Hmm wait, let me re-check: could this actually be intentional and correct? The carry for a chunk is state at chunk start (belonging to seq A). For tokens in seq B within that chunk, correct answer is local only. The code adds prefix*carry. Bug.

But wait — check the Mamba original: in chunked scan, seq boundaries reset. Here in original mamba code they'd zero the inter-chunk carry across boundaries similarly. So this is a real defect for mid-chunk sequence starts.

But wait, maybe I misread pass_states. factor zeroed if next_sequence != sequence. That handles chunk-boundary starts only. Mid-chunk starts not handled.

Also check _combine prefix masking: prefix loaded with mask rows<L other=0. previous loaded at chunk*K-1 (last token of previous chunk) with mask chunk>0 other=0; for chunk 0 previous=0, and current at row 0 is 0 (row starts at label 0), so prefix kept for chunk 0 — carry = initial state, correct.

Edge case: L not multiple of K. Pass_states: representative = chunk*K may be >= L for last chunk; tl.load(IDS + batch*L + representative) — out-of-bounds load! For the last chunk with begin >= L, loading IDS at index representative which can exceed L-1. That's an OOB read. Actually C = cdiv(L, K), last chunk begin = (C-1)*K < L always (since chunks cover L). chunk*K < L for all chunks < C? Yes, since chunk C-1: (C-1)*K < L by definition of cdiv. So representative < L, fine.

Also factor for tokens beyond L in last chunk: handled by valid mask.

Mid-chunk boundary bug: let me verify with a concrete case. Chunk of size K, seq boundary at token t within chunk (1 <= j < K-1). Token after boundary (j+1..): local computed correctly from zero. prefix[j] nonzero (product of alphas incl. across boundary). _combine: prefix zeroed only at j. So result[j+1] = local + prefix*carry — wrong unless carry is zero or alphas... carry is state at chunk start from previous sequence — generally nonzero. Error magnitude: prefix <= 0.96875^K... could be tiny? 0.96875^32 ≈ 0.36, times carry <= 1... error could be up to ~0.36, exceeding tolerance 0.002. Actually prefix at token j+1 includes at least one alpha ≥ 0.5, carry ≤ ... hmm could still exceed tolerance.

Actually wait — also factor in pass_states: it multiplies carry by factor; carry at chunk start belongs to a sequence that may have ended mid-chunk; that carry is still correct as state at chunk start. Fine.

So the defect: mid-chunk sequence restarts add spurious carry. Reject.

But hold on — maybe prefix should have been reset in _local_chunks at boundaries. It wasn't. Also check: what if boundary is exactly at chunk start? Then previous (in _combine, chunk's first token label != chunk's last token label of previous chunk) → prefix zeroed; also pass_states factor zeroed. Handled. Only mid-chunk boundary broken. That's a real defect within domain (labels nondecreasing, new sequences can start anywhere).

Also note in _local_chunks, prefix product isn't reset at boundary — even at chunk start boundary it's computed but zeroed later. OK.

Verdict: reject.

## Final response (verbatim)

{"verdict":"reject","confidence":0.85,"reason":"When a new sequence starts mid-chunk, _local_chunks resets the local state to zero but keeps the cumulative alpha product, and _combine only zeroes that prefix at the exact boundary token; subsequent tokens of the new sequence therefore incorrectly add prefix*carry (carry being the chunk-start state from the previous sequence), which can exceed the 0.002 tolerance. Only boundaries aligned with chunk starts are handled correctly (via _pass_states' factor reset and _combine's previous-token check), so the implementation is wrong for the general packed-sequence domain."}

## Verdict

{
  "verdict": "reject",
  "confidence": 0.85,
  "reason": "When a new sequence starts mid-chunk, _local_chunks resets the local state to zero but keeps the cumulative alpha product, and _combine only zeroes that prefix at the exact boundary token; subsequent tokens of the new sequence therefore incorrectly add prefix*carry (carry being the chunk-start state from the previous sequence), which can exceed the 0.002 tolerance. Only boundaries aligned with chunk starts are handled correctly (via _pass_states' factor reset and _combine's previous-token check), so the implementation is wrong for the general packed-sequence domain."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "reject",
    "confidence": 0.85,
    "reason": "When a new sequence starts mid-chunk, _local_chunks resets the local state to zero but keeps the cumulative alpha product, and _combine only zeroes that prefix at the exact boundary token; subsequent tokens of the new sequence therefore incorrectly add prefix*carry (carry being the chunk-start state from the previous sequence), which can exceed the 0.002 tolerance. Only boundaries aligned with chunk starts are handled correctly (via _pass_states' factor reset and _combine's previous-token check), so the implementation is wrong for the general packed-sequence domain."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 3444,
    "output_tokens": 1638
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 31.39677945803851,
  "kernel_sha256": "6f81a6a9bca854e05b71af4089357ccc53ff359aed3d466cf1bc77d9f139295e",
  "problem_sha256": "151b9d9ce0b44decc5bd9838f457db00e92b1d8cef808a22aa66a85e76b361c9",
  "prompt_variant": "original",
  "estimated_usd": 0.0120288,
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
