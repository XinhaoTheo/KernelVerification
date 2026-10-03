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
        representative = tl.minimum((chunk + 1) * K, L) - 1
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

Let me analyze the chunked recurrence.

Chunk-local: state starts at 0 within each chunk, but resets on sequence boundary (prior = 0 when current != previous). prefix = product of alphas (unconditionally multiplies alpha even when invalid token? mask other=1.0, so fine; also for sequence reset—prefix is product of all alphas, but if a sequence boundary occurs mid-chunk, the carry-in shouldn't be multiplied past the boundary). Wait, in _pass_states: factor = product of alphas over chunk; but if sequence boundary inside chunk, incoming carry shouldn't propagate past boundary. They handle it via `representative`: next_sequence at last token of chunk vs sequence at start (which is the sequence of the chunk's last token's previous... hmm).

In _pass_states, `sequence` tracks the sequence label of the last token of the previous chunk (or 0 initially... actually state is initial_state, sequence=0). Then for each chunk, factor is zeroed if next_sequence != sequence, i.e., a boundary occurs somewhere between previous chunk's last token and this chunk's last token? That's not precise — if a boundary occurs inside this chunk, factor should be 0. If boundary occurs inside this chunk, then sequence at this chunk's last token differs from sequence at previous chunk's last token? Yes: labels nondecreasing; if a boundary occurs within the chunk, last token's label > previous chunk's last token's label. If no boundary in this chunk, labels equal. But what if boundary occurs exactly between chunks (i.e., at first token of this chunk)? Then previous chunk's last label < this chunk's last label → factor zeroed, correct since new sequence starts at this chunk's first token. Also state reset: state = factor*state + new_state where new_state is the chunk-local summary starting from zero at boundary — correct since local resets at boundary.

But wait: sequence var starts at 0 but state is initial_state — what if the first token's label... labels start at 0, so fine.

Also: sequence zeroing factor resets carry at boundary — but what if multiple boundaries in a chunk? Last-label comparison still detects it. And local chunk resets at each boundary. Good.

_combine: prefix[row] = product of alphas from chunk start to row inclusive, zeroed if current != previous (previous = label of token rows[0]-1, i.e., chunk start -1, other=0 when chunk==0; for chunk 0, previous=0 and current at row0 is 0, ok... wait mask=chunk>0, other=0; first token's label is 0, so equal). Then prefix[:, None]*incoming + local. Local at row includes alphas up to that row applied within chunk with resets. Incoming is carry before chunk. prefix is product of alphas from chunk start through that row. Correct — but if a boundary occurs mid-chunk at row r, prefix at rows >= r is zeroed? No! prefix is only zeroed at row where current != previous-of-chunk-start. Hmm: previous is only chunk*K-1 (single value), not per-row. So prefix[row] = full product from chunk start, zeroed only if row r0's label differs from chunk-start-1 label.

Problem: consider boundary at row r inside chunk. Then rows >= r should have prefix contribution 0 (new sequence, no carry). But the code: prefix[row] = product of alphas from chunk start to row, and zeroed only if rows[0]'s label != previous. For rows >= r, the carry contribution should be zero. The code gives product over alphas including those before the boundary — nonzero! Bug?

Wait, but LOCAL already contains local state that resets at boundary. The incoming contribution for rows >= boundary should be 0. The code computes prefix as product of all alphas in chunk up to row — for rows after the boundary, prefix * incoming ≠ 0 incorrectly.

Hmm, but wait — _pass_states zeroed factor for chunks containing boundaries, so carry into that chunk... no, carry into boundary-containing chunk is zeroed only at that chunk. For a chunk with no boundary at start but boundary at row r: carry into chunk is initial/carry, correct for rows < r. For rows >= r, output should be local-only (from boundary). But result = local + prefix*incoming where prefix includes alphas before boundary — wrong!

Unless prefix should be re-zeroed at boundary. Let me re-check: `prefix = tl.where(current == previous, prefix, 0.0)` — previous is scalar (chunk*K-1 label), current is per-row. Only row 0 is compared? No — current is a vector over rows; previous is scalar. So prefix[row] is zeroed where current[row] != label at chunk*K-1. For rows after a boundary, current[row] > that label, so zeroed! Yes — since boundary inside chunk means labels at rows >= r differ from label at chunk*K-1 (nondecreasing labels, boundary increases label). So prefix is zeroed for all rows at/after the first boundary in the chunk. Correct.

Edge: boundary at row r but label equal? Labels strictly increase at boundary, so fine. What about multiple chunks — carry stored per chunk before update: carry[chunk] = state before applying chunk. Yes, stored at loop start. Combine uses carry[chunk]. Good.

What about token 0 of batch: initial_state used via _pass_states starting state = initial. carry[0] = initial. prefix[0] zeroed? chunk 0, previous = load masked other=0, current[0]=0 → equal, prefix kept. local[0] = u[0] (state starts 0, prior: current==previous? previous= IDS[0] initially loaded = label at begin=0, which is 0; current row 0 = 0, equal → prior = state = 0). Output[0] = u[0] + prefix[0]*initial. prefix[0] = alpha[0] = decay[0]. Correct: decay[0]*initial + u[0].

Invalid tokens (token >= L): alpha other=1.0, prefix stays, but stores masked. prefix stored masked by valid — fine. But prefix for invalid rows doesn't matter. However within chunk, prefix accumulates over invalid tokens as 1.0 — fine.

In _pass_states, representative = min((chunk+1)*K, L)-1 — last valid token. next_sequence vs `sequence` initialized 0: for chunk 0, last token label could be >0 if boundary inside first chunk → factor zeroed. Correct, since carry into chunk 0 is initial which applies only to first sequence.

Edge: chunk containing boundary where the new sequence's carry should be 0 — handled. But what if boundary is at last token of a chunk? E.g., labels: chunk1 ends with label 1 starting at last token. Then chunk 2's factor: next_sequence(chunk2 last) vs sequence(chunk1 last =1): if chunk2 has no boundary, equal → factor kept, carry = state after chunk1 which correctly started new sequence... carry propagates across since same sequence. Correct.

Wait subtle: `_pass_states` factor zeroing checks boundary between end of previous chunk and end of current chunk. If boundary occurs at first token of current chunk, last labels differ → factor zero → correct (carry shouldn't propagate). If boundary in middle of current chunk → zeroed, correct. If no boundary → kept, correct. But boundary could occur at first token of current chunk AND... labels only increase, fine.

Hmm, one more: factor zeroing when boundary at the very first token of a new sequence in current chunk — new_state (local summary) starts from 0 at boundary — correct.

What about sequence variable `sequence` initialization: 0, but state = initial. For chunk 0, if chunk 0's last token label is 0 (no boundary in row), factor kept → state = prefix_all * initial + local. Correct.

FP32 accumulation vs tolerance: chunk products of decay^32 ~ 0.96875^32 ≈ 0.36, fine. Tolerance 0.002+1e-4|target|, magnitudes ≤ ~32. Prefix products computed in FP32 per-chunk (≤64 tokens), fine.

`prefix *= alpha` unmasked even when token invalid — alpha other=1.0. OK.

One concern: `tl.load(IDS + batch * L + chunk * K - 1, mask=chunk > 0, other=0)` — for chunk==0, address computed chunk*K-1 = -1, masked, fine.

`_combine` loads prefix masked rows<L other=0, then where current==previous... current other=-1, so invalid rows prefix=0, stored masked anyway.

`enable_fp_fusion=False` fine.

BD next_power_of_2(width): loads masked cols<D. Stores masked. Good.

Final state: state after all chunks in _pass_states. For invalid tokens beyond L: chunks = cdiv(L,K), last chunk partially valid; summary/scale stored for full chunk (mask cols<D only, stores not gated by validity of tokens — summary state only includes valid tokens since state updates masked by valid). scale = prefix product includes alpha=1.0 for invalid → correct. But scale is used in _pass_states to zero factor if boundary in last chunk; representative = L-1 handled. Final = state correct.

Also note: prefix store in _local_chunks masked valid, but invalid region of prefix tensor read in _combine masked. fine.

So looks correct. One more check: _local_chunks `previous` initial load at begin = chunk*K (no mask, begin < L always since chunks = cdiv(L,K) and chunk<C → begin<L? begin = (C-1)*K < L yes). Fine.

I think trust. Confidence high.

## Final response (verbatim)

{"verdict": "trust", "confidence": 0.9, "reason": "The three-stage chunked scheme is mathematically consistent: chunk-local states reset at sequence boundaries, the carry pass zeroes the scale factor when a boundary occurs between chunk ends (detected via last-token labels), and _combine zeroes the prefix contribution for rows at/after the first boundary in a chunk (since labels are nondecreasing, current != label at chunk*K-1 identifies them), so cross-boundary carry is correctly excluded. Boundary conditions (chunk 0, invalid tail tokens, initial state at t=0) are all handled with proper masking, and FP32 accumulation over chunks of ≤64 tokens stays well within the stated tolerance."}

## Verdict

{
  "verdict": "trust",
  "confidence": 0.9,
  "reason": "The three-stage chunked scheme is mathematically consistent: chunk-local states reset at sequence boundaries, the carry pass zeroes the scale factor when a boundary occurs between chunk ends (detected via last-token labels), and _combine zeroes the prefix contribution for rows at/after the first boundary in a chunk (since labels are nondecreasing, current != label at chunk*K-1 identifies them), so cross-boundary carry is correctly excluded. Boundary conditions (chunk 0, invalid tail tokens, initial state at t=0) are all handled with proper masking, and FP32 accumulation over chunks of ≤64 tokens stays well within the stated tolerance."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "trust",
    "confidence": 0.9,
    "reason": "The three-stage chunked scheme is mathematically consistent: chunk-local states reset at sequence boundaries, the carry pass zeroes the scale factor when a boundary occurs between chunk ends (detected via last-token labels), and _combine zeroes the prefix contribution for rows at/after the first boundary in a chunk (since labels are nondecreasing, current != label at chunk*K-1 identifies them), so cross-boundary carry is correctly excluded. Boundary conditions (chunk 0, invalid tail tokens, initial state at t=0) are all handled with proper masking, and FP32 accumulation over chunks of ≤64 tokens stays well within the stated tolerance."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 3457,
    "output_tokens": 2252
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 43.68690945813432,
  "kernel_sha256": "fb33bac7b01c84150298b037b67a717ad1406880f25189df562cbfc72720389f",
  "problem_sha256": "151b9d9ce0b44decc5bd9838f457db00e92b1d8cef808a22aa66a85e76b361c9",
  "prompt_variant": "original",
  "estimated_usd": 0.0147486,
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
