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

Verify a CUDA sliding-window state-update implementation over the complete
input domain below, not only the example returned by make_inputs().

Inputs are contiguous FP32 CUDA tensors on the same NVIDIA T4 device:
- state has shape [B, L, D], where 1 <= B <= 4, 2 <= L <= 4096,
  and 1 <= D <= 1024.
- new_events has shape [B, K, D], where 1 <= K <= min(L, 128).
- The two input tensors have disjoint storage. Their values may be any finite
  FP32 values, including zeros and signed zeros.

Let old_state be the contents of state before this call. After run() completes,
for every legal batch b, row r, and feature d:
  state[b,r,d] = old_state[b,r+K,d]  if r < L-K
                 new_events[b,r-(L-K),d] otherwise.
Each result must preserve the exact FP32 bit pattern of its source element.
This is a copy operation; no numerical approximation is permitted.

run() must update state in place and return that same tensor (same storage),
while preserving new_events. Calls may be repeated on the state resulting from
previous calls, with any legal new_events. All calls are on the same CUDA stream,
with no concurrent external access to either tensor. CUDA's usual asynchronous
return is allowed; correctness is observed after stream synchronization.

The full wrapper, including its allocation decisions and launch arguments, is
part of the implementation to verify. Do not assume distinct pointer names in
the Triton function imply distinct storage: inspect the arguments passed by run.
The public reference() in kernel.py computes the required new contents without
mutating either input and can also execute on CPU. Evaluate it BEFORE run() or
on a saved pre-call state. You may construct any legal inputs and repeat calls.
The frozen environment is PyTorch 2.8.0 / Triton 3.4.0 / NVIDIA T4.


## Kernel implementation under test (kernel.py)

```python
"""Standalone CUDA sliding-window state update.

The public contract and independent reference are in this file and problem.txt.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _slide(STATE, EVENTS, OUT, L: tl.constexpr, D: tl.constexpr,
           K: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    batch = index // (L * D)
    row = (index // D) % L
    col = index % D
    from_state = row < L - K
    old_value = tl.load(STATE + batch * L * D + (row + K) * D + col,
                        mask=valid & from_state, other=0.0)
    new_value = tl.load(EVENTS + batch * K * D + (row - (L - K)) * D + col,
                        mask=valid & ~from_state, other=0.0)
    value = tl.where(from_state, old_value, new_value)
    tl.store(OUT + index, value, mask=valid)


@triton.jit
def _writeback(SOURCE, STATE, TOTAL: tl.constexpr,
               NEGATE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = index < TOTAL
    source_bits = SOURCE.to(tl.pointer_type(tl.uint32))
    state_bits = STATE.to(tl.pointer_type(tl.uint32))
    bits = tl.load(source_bits + index, mask=mask, other=0)
    if NEGATE:
        bits = bits ^ 0x80000000
    tl.store(state_bits + index, bits, mask=mask)


def run(state, new_events):
    """Update state in place, return that same tensor, preserve new_events."""
    assert state.is_cuda and new_events.is_cuda
    assert state.device == new_events.device
    assert state.dtype == new_events.dtype == torch.float32
    assert state.is_contiguous() and new_events.is_contiguous()
    assert state.ndim == new_events.ndim == 3
    batch, length, width = state.shape
    assert new_events.shape[0] == batch and new_events.shape[2] == width
    steps = new_events.shape[1]
    assert 1 <= batch <= 4 and 2 <= length <= 4096 and 1 <= width <= 1024
    assert 1 <= steps <= min(length, 128)
    with torch.no_grad():
        source = state.resolve_neg()
        events_source = new_events.resolve_neg()
    output = torch.empty_like(state)
    _slide[(triton.cdiv(state.numel(), 1024),)](
        source, events_source, output, length, width, steps, state.numel(),
        BLOCK=1024, num_warps=4,
    )
    _writeback[(triton.cdiv(state.numel(), 1024),)](
        output, state, state.numel(), NEGATE=state.is_neg(), BLOCK=1024, num_warps=4,
    )
    return state


def reference(state, new_events):
    """Independent result computed from the pre-call state, without mutation."""
    return torch.cat((state[:, new_events.shape[1]:, :], new_events), dim=1)


def make_inputs(device="cuda", batch=4, length=2048, width=1024,
                steps=2, seed=0):
    """One convenient example; any legal input may be tested."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn((batch, length, width), generator=generator)
    new_events = torch.randn((batch, steps, width), generator=generator)
    return state.to(device), new_events.to(device)

```

Is this implementation correct?

## Provider reasoning (verbatim)

Slide kernel: index decomposition correct. For rows >= L-K, load EVENTS at row-(L-K), 0..K-1 — correct. Writes to OUT then writeback copies bits, negate handling. In-place writeback from output (separate tensor), fine. Bit-exact via uint32 copy. Signed zeros preserved. state.resolve_neg() gives base tensor; correct handling of negative view? The output then negated back — correct.

One issue: from_state load offset row+K up to L-1 — fine. Negation: if state is negative view, _slide reads base tensor (positive values), then writeback negates — correct. Looks correct.

## Final response (verbatim)

{"verdict": "trust", "confidence": 0.9, "reason": "The slide kernel correctly maps each output index to old_state[b,row+K] for row < L-K and new_events[b,row-(L-K)] otherwise, writing into a separate buffer that is then copied bit-exactly (uint32) back into state's storage, handling negative views via resolve_neg/is_neg and preserving bit patterns including signed zeros."}

## Verdict

{
  "verdict": "trust",
  "confidence": 0.9,
  "reason": "The slide kernel correctly maps each output index to old_state[b,row+K] for row < L-K and new_events[b,row-(L-K)] otherwise, writing into a separate buffer that is then copied bit-exactly (uint32) back into state's storage, handling negative views via resolve_neg/is_neg and preserving bit patterns including signed zeros."
}

## Usage and stop reason

{
  "model": "accounts/fireworks/models/glm-5p3",
  "provider": "fireworks",
  "response": {
    "verdict": "trust",
    "confidence": 0.9,
    "reason": "The slide kernel correctly maps each output index to old_state[b,row+K] for row < L-K and new_events[b,row-(L-K)] otherwise, writing into a separate buffer that is then copied bit-exactly (uint32) back into state's storage, handling negative views via resolve_neg/is_neg and preserving bit patterns including signed zeros."
  },
  "reasoning_effort": "low",
  "timeout_s": 1800,
  "usage": {
    "input_tokens": 1553,
    "output_tokens": 229
  },
  "stop_reason": "stop",
  "max_tokens": 32768,
  "elapsed_s": 6.731528665870428,
  "kernel_sha256": "ef44103fef75c3d21c41cbbfa0155a9e05b3082c09ef0c4d425140b75d2e66ef",
  "problem_sha256": "796219b618572fe1749cd02af33d89b699dea3413ed3fd8d2ba1fdb2c3797496",
  "prompt_variant": "original",
  "estimated_usd": 0.0031818000000000003,
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
